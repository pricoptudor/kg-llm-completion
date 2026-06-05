"""LLM scorer for KG completion via per-candidate log-probability.

Plugs a causal language model into the SAME `Scorer` protocol the KGE models use,
so the LLM is judged by the identical filtered-ranking harness.

For a tail query (h, r, ?) we build a plain completion prompt from readable names,
e.g. "Head: Albert Einstein. Relation: place of birth. Tail:", and score each
candidate entity `e` by the model's length-normalized log-probability of e's name
as the continuation:

    score(e) = (1/|tokens(e)|) * sum_i log p(token_i | prompt + earlier tokens)

Higher (closer to 0) = more plausible. We rank entities by this score and feed it
to evaluate() exactly like a KGE score matrix.

Efficiency — KV-cache prefix reuse (default):
  Every candidate of a query shares the same prompt prefix. We run the prefix
  through the model ONCE, keep its attention K/V cache, and then for each candidate
  batch we only forward the short answer tokens on top of that cache. The prefix is
  the bulk of the tokens, so this is a large saving versus re-encoding it per
  candidate. Set `use_prefix_cache=False` to fall back to the simple (slower but
  maximally compatible) re-encode-everything path.

Length normalization (mean log-prob per token, default) removes the bias against
multi-token entity names; `length_normalize=False` gives the raw sum.
"""

from __future__ import annotations

import torch

from kg_llm.data.fb15k237 import FB15k237


def tail_prompt(head_name: str, relation_name: str) -> str:
    return f"Head: {head_name}. Relation: {relation_name}. Tail:"


def head_prompt(tail_name: str, relation_name: str) -> str:
    return f"Tail: {tail_name}. Relation: {relation_name}. Head:"


class LLMScorer:
    """Wrap a HF causal LM so it satisfies `kg_llm.eval.ranking.Scorer`."""

    def __init__(
        self,
        model,
        tokenizer,
        dataset: FB15k237,
        *,
        length_normalize: bool = True,
        cand_batch_size: int = 128,
        use_prefix_cache: bool = True,
        device=None,
    ) -> None:
        self.model = model.eval()
        self.tok = tokenizer
        self.ds = dataset
        self.length_normalize = length_normalize
        self.cand_batch_size = cand_batch_size
        self.use_prefix_cache = use_prefix_cache
        self.device = device or next(model.parameters()).device

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        # Answer tokens for every entity name, precomputed once. A leading space
        # makes the name tokenize as a continuation (" Ulm", matching its place
        # after the prompt's trailing colon).
        self._cand_tokens: list[list[int]] = [
            self.tok(" " + dataset.entity_name(i), add_special_tokens=False).input_ids
            for i in range(dataset.num_entities)
        ]

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def _score_against_prefix(self, prefix: str) -> torch.Tensor:
        if self.use_prefix_cache:
            return self._score_cached(prefix)
        return self._score_batched(prefix)

    @torch.no_grad()
    def _score_cached(self, prefix: str) -> torch.Tensor:
        """Prefix encoded once; only answer tokens are forwarded per candidate."""
        prefix_ids = self.tok(prefix, add_special_tokens=True).input_ids
        plen = len(prefix_ids)
        E = self.ds.num_entities
        scores = torch.full((E,), float("-inf"))
        dev = self.device

        pref_out = self.model(
            input_ids=torch.tensor([prefix_ids], device=dev), use_cache=True
        )
        cache = pref_out.past_key_values
        if hasattr(cache, "to_legacy_cache"):  # transformers Cache object -> tuples
            cache = cache.to_legacy_cache()
        # Logit from the last prefix position predicts the FIRST answer token.
        first_lp = torch.log_softmax(pref_out.logits[0, -1, :].float(), dim=-1)

        for start in range(0, E, self.cand_batch_size):
            cand = list(range(start, min(start + self.cand_batch_size, E)))
            ans_lens = [len(self._cand_tokens[i]) for i in cand]
            max_a = max(max(ans_lens), 1)
            B = len(cand)

            ans_ids = torch.full((B, max_a), self.tok.pad_token_id, dtype=torch.long)
            for j, i in enumerate(cand):
                t = self._cand_tokens[i]
                if t:
                    ans_ids[j, : len(t)] = torch.tensor(t, dtype=torch.long)
            ans_ids = ans_ids.to(dev)

            # Expand the (batch-1) prefix cache across the candidate batch.
            past = tuple(
                (k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous())
                for (k, v) in cache
            )
            # Attention covers [prefix (all real) | answer (real up to its length)].
            attn = torch.ones((B, plen + max_a), dtype=torch.long, device=dev)
            for j, a in enumerate(ans_lens):
                if a < max_a:
                    attn[j, plen + a :] = 0
            position_ids = torch.arange(plen, plen + max_a, device=dev).unsqueeze(0).expand(B, -1)

            out = self.model(
                input_ids=ans_ids,
                attention_mask=attn,
                past_key_values=past,
                position_ids=position_ids,
                use_cache=False,
            )
            # out.logits[:, m, :] is produced after consuming answer token m, so it
            # predicts answer token m+1.
            ans_lp = torch.log_softmax(out.logits.float(), dim=-1)

            for j, i in enumerate(cand):
                toks = self._cand_tokens[i]
                a = len(toks)
                if a == 0:
                    continue
                lp = first_lp[toks[0]]
                if a > 1:
                    pos = torch.arange(a - 1, device=dev)
                    rest = torch.tensor(toks[1:], device=dev)
                    lp = lp + ans_lp[j, pos, rest].sum()
                scores[i] = (lp / a if self.length_normalize else lp).item()

        return scores

    @torch.no_grad()
    def _score_batched(self, prefix: str) -> torch.Tensor:
        """Fallback: re-encode prefix+answer for every candidate (max compatibility)."""
        prefix_ids = self.tok(prefix, add_special_tokens=True).input_ids
        plen = len(prefix_ids)
        E = self.ds.num_entities
        scores = torch.full((E,), float("-inf"))
        dev = self.device

        for start in range(0, E, self.cand_batch_size):
            cand = list(range(start, min(start + self.cand_batch_size, E)))
            seqs = [prefix_ids + self._cand_tokens[i] for i in cand]
            ans_lens = [len(self._cand_tokens[i]) for i in cand]
            max_a = max(max(ans_lens), 1)
            maxlen = plen + max_a

            input_ids = torch.full((len(seqs), maxlen), self.tok.pad_token_id, dtype=torch.long)
            attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
            for j, s in enumerate(seqs):
                input_ids[j, : len(s)] = torch.tensor(s, dtype=torch.long)
                attn[j, : len(s)] = 1
            input_ids, attn = input_ids.to(dev), attn.to(dev)

            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            sub_lp = torch.log_softmax(logits[:, plen - 1 : plen - 1 + max_a, :].float(), dim=-1)

            for j, i in enumerate(cand):
                toks = self._cand_tokens[i]
                a = len(toks)
                if a == 0:
                    continue
                pos = torch.arange(a, device=dev)
                t = torch.tensor(toks, device=dev)
                lp = sub_lp[j, pos, t].sum()
                scores[i] = (lp / a if self.length_normalize else lp).item()

        return scores

    # ------------------------------------------------------------- Scorer API
    @torch.no_grad()
    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        out = torch.empty(len(heads), self.ds.num_entities)
        for k in range(len(heads)):
            prefix = tail_prompt(
                self.ds.entity_name(int(heads[k])), self.ds.relation_name(int(relations[k]))
            )
            out[k] = self._score_against_prefix(prefix)
        return out

    @torch.no_grad()
    def score_heads(self, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        out = torch.empty(len(relations), self.ds.num_entities)
        for k in range(len(relations)):
            prefix = head_prompt(
                self.ds.entity_name(int(tails[k])), self.ds.relation_name(int(relations[k]))
            )
            out[k] = self._score_against_prefix(prefix)
        return out
