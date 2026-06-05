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

On efficiency: we score candidates in batches (cand_batch_size at a time), each a
single forward pass. We do NOT hand-roll prefix KV-cache reuse — it isn't portable
across architectures (e.g. Qwen3.5 uses linear attention, which has no expandable
key/value cache) or across transformers' evolving Cache API. If full-test runs ever
need to be faster, the right tool is vLLM, which does prefix caching internally.

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
        device=None,
    ) -> None:
        self.model = model.eval()
        self.tok = tokenizer
        self.ds = dataset
        self.length_normalize = length_normalize
        self.cand_batch_size = cand_batch_size
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

    @torch.no_grad()
    def _score_against_prefix(self, prefix: str) -> torch.Tensor:
        """Return a (num_entities,) tensor: log-prob score of every entity name."""
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
            # Answer token at sequence position p is predicted by logits at p-1.
            # All candidates share the prefix, so answer positions start at plen-1.
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
