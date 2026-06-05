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

Performance notes:
- Candidate names are tokenized ONCE into a padded matrix at init.
- Scoring is fully vectorized per candidate batch: one forward pass, one gather,
  one masked sum. We never call .item() inside the candidate loop — doing so forces
  a GPU->CPU sync per candidate (~14.5k per query) and was a ~30x slowdown.
- We do not hand-roll prefix KV-cache reuse: it isn't portable across architectures
  (Qwen3.5 uses linear attention) or transformers' Cache API. For faster full-test
  runs, the right tool is vLLM (internal prefix caching).
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

        # Tokenize every entity name once (leading space => continuation tokens),
        # then pack into a padded (num_entities, Lmax) matrix plus a length mask.
        cand_tokens = [
            self.tok(" " + dataset.entity_name(i), add_special_tokens=False).input_ids
            for i in range(dataset.num_entities)
        ]
        E = dataset.num_entities
        lmax = max((len(t) for t in cand_tokens), default=1) or 1

        matrix = torch.zeros((E, lmax), dtype=torch.long)
        mask = torch.zeros((E, lmax))
        lengths = torch.zeros(E, dtype=torch.long)
        for i, t in enumerate(cand_tokens):
            a = len(t)
            if a:
                matrix[i, :a] = torch.tensor(t, dtype=torch.long)
                mask[i, :a] = 1.0
            lengths[i] = a

        self._cand_matrix = matrix.to(self.device)
        self._cand_mask = mask.to(self.device)
        self._cand_len = lengths.to(self.device)

    @torch.no_grad()
    def _score_against_prefix(self, prefix: str) -> torch.Tensor:
        """Return a (num_entities,) tensor: log-prob score of every entity name."""
        prefix_ids = self.tok(prefix, add_special_tokens=True).input_ids
        plen = len(prefix_ids)
        E = self.ds.num_entities
        dev = self.device
        prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=dev)
        scores = torch.empty(E, device=dev)

        for start in range(0, E, self.cand_batch_size):
            end = min(start + self.cand_batch_size, E)
            B = end - start
            lens = self._cand_len[start:end]
            max_a = int(lens.max())
            if max_a == 0:
                scores[start:end] = float("-inf")
                continue

            cand_block = self._cand_matrix[start:end, :max_a]  # (B, max_a)
            mask_block = self._cand_mask[start:end, :max_a]  # (B, max_a)

            input_ids = torch.cat([prefix_t.unsqueeze(0).expand(B, plen), cand_block], dim=1)
            attn = torch.cat([torch.ones(B, plen, device=dev), mask_block], dim=1).long()

            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            # Answer token at sequence position p is predicted by logits at p-1.
            # All candidates share the prefix, so answer positions start at plen-1.
            sub_lp = torch.log_softmax(logits[:, plen - 1 : plen - 1 + max_a, :].float(), dim=-1)

            # Gather the log-prob of each candidate's own answer tokens, mask pads.
            tok_lp = sub_lp.gather(2, cand_block.unsqueeze(-1)).squeeze(-1)  # (B, max_a)
            summed = (tok_lp * mask_block).sum(dim=1)  # (B,)
            if self.length_normalize:
                summed = summed / lens.clamp(min=1).float()
            scores[start:end] = summed

        return scores.cpu()

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
