"""LLM scorer for KG completion via per-candidate log-probability.

Plugs a causal language model into the SAME `Scorer` protocol the KGE models use,
so the LLM is judged by the identical filtered-ranking harness.

For a tail query (h, r, ?) we build a plain completion prompt from readable names,
e.g. "Head: Albert Einstein. Relation: place of birth. Tail:", and score each
candidate entity `e` by the model's length-normalized log-probability of e's name
as the continuation:

    score(e) = (1/|tokens(e)|) * sum_i log p(token_i | prompt + earlier tokens)

Higher (closer to 0) = more plausible.

Scoring ALL 14,541 entities per query is ~14.5k short forwards per query, which is
infeasible on a T4 for more than a handful of queries. So in practice we use
`evaluate_llm_sampled`: rank the gold against a sampled set of negatives. The
LLMScorer exposes `score_*_candidates` to score an arbitrary entity-id subset.

Implementation notes:
- Candidate names are tokenized ONCE into a padded matrix at init.
- Scoring is vectorized per candidate batch (one forward, gather, masked sum); we
  never call .item() in the hot path (that forces a GPU->CPU sync per candidate).
- No hand-rolled KV-cache reuse (not portable: Qwen3.5 is linear-attention). For a
  faster full eval later, use vLLM (internal prefix caching).
"""

from __future__ import annotations

import torch
from tqdm.auto import tqdm

from kg_llm.data.fb15k237 import FB15k237


def tail_prompt(head_name: str, relation_name: str) -> str:
    return f"Head: {head_name}. Relation: {relation_name}. Tail:"


def head_prompt(tail_name: str, relation_name: str) -> str:
    return f"Tail: {tail_name}. Relation: {relation_name}. Head:"


class LLMScorer:
    """Wrap a HF causal LM so it can score candidate entity names for a query."""

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
    def _score_indices(self, prefix: str, idx: torch.Tensor) -> torch.Tensor:
        """Score the entities in `idx` (1-D long) as continuations of `prefix`.

        Returns a (len(idx),) tensor aligned to `idx`.
        """
        dev = self.device
        idx = idx.to(dev)
        prefix_ids = self.tok(prefix, add_special_tokens=True).input_ids
        plen = len(prefix_ids)
        prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=dev)
        n = idx.shape[0]
        out = torch.empty(n, device=dev)

        for start in range(0, n, self.cand_batch_size):
            chunk = idx[start : start + self.cand_batch_size]
            B = chunk.shape[0]
            lens = self._cand_len[chunk]
            max_a = int(lens.max())
            if max_a == 0:
                out[start : start + B] = float("-inf")
                continue
            cand_block = self._cand_matrix[chunk][:, :max_a]
            mask_block = self._cand_mask[chunk][:, :max_a]

            input_ids = torch.cat([prefix_t.unsqueeze(0).expand(B, plen), cand_block], dim=1)
            attn = torch.cat([torch.ones(B, plen, device=dev), mask_block], dim=1).long()

            logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            sub_lp = torch.log_softmax(logits[:, plen - 1 : plen - 1 + max_a, :].float(), dim=-1)
            tok_lp = sub_lp.gather(2, cand_block.unsqueeze(-1)).squeeze(-1)
            summed = (tok_lp * mask_block).sum(dim=1)
            if self.length_normalize:
                summed = summed / lens.clamp(min=1).float()
            out[start : start + B] = summed

        return out.cpu()

    # candidate-subset scoring (used by evaluate_llm_sampled)
    def score_tail_candidates(self, head_id, relation_id, candidate_ids) -> torch.Tensor:
        prefix = tail_prompt(self.ds.entity_name(int(head_id)), self.ds.relation_name(int(relation_id)))
        return self._score_indices(prefix, torch.as_tensor(candidate_ids, dtype=torch.long))

    def score_head_candidates(self, relation_id, tail_id, candidate_ids) -> torch.Tensor:
        prefix = head_prompt(self.ds.entity_name(int(tail_id)), self.ds.relation_name(int(relation_id)))
        return self._score_indices(prefix, torch.as_tensor(candidate_ids, dtype=torch.long))

    # full Scorer protocol (scores ALL entities — exact but expensive; small sets only)
    @torch.no_grad()
    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        allidx = torch.arange(self.ds.num_entities)
        out = torch.empty(len(heads), self.ds.num_entities)
        for k in range(len(heads)):
            prefix = tail_prompt(self.ds.entity_name(int(heads[k])), self.ds.relation_name(int(relations[k])))
            out[k] = self._score_indices(prefix, allidx)
        return out

    @torch.no_grad()
    def score_heads(self, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        allidx = torch.arange(self.ds.num_entities)
        out = torch.empty(len(relations), self.ds.num_entities)
        for k in range(len(relations)):
            prefix = head_prompt(self.ds.entity_name(int(tails[k])), self.ds.relation_name(int(relations[k])))
            out[k] = self._score_indices(prefix, allidx)
        return out


def _sample_negatives(num_entities: int, k: int, exclude: set[int], gen: torch.Generator) -> list[int]:
    """k distinct entity ids in [0, num_entities) not in `exclude`."""
    perm = torch.randperm(num_entities, generator=gen).tolist()
    out = []
    for x in perm:
        if x not in exclude:
            out.append(x)
            if len(out) == k:
                break
    return out


def _realistic_rank(scores: torch.Tensor) -> float:
    """Rank of the gold (index 0) among `scores`, with realistic tie handling."""
    gold = scores[0]
    others = scores[1:]
    greater = int((others > gold).sum())
    tied = int((others == gold).sum())
    return 1.0 + greater + 0.5 * tied


@torch.no_grad()
def evaluate_llm_sampled(
    scorer: LLMScorer,
    triples: torch.Tensor,
    filtered_index,
    num_entities: int,
    *,
    num_candidates: int = 256,
    seed: int = 0,
    ks: tuple[int, ...] = (1, 3, 10),
):
    """Filtered ranking of the gold against `num_candidates-1` sampled negatives.

    Pools head and tail queries, like the full harness. Negatives exclude all
    known-true answers (the filtered protocol) and the gold. NOTE: metrics are over
    `num_candidates` candidates, NOT all 14,541 — comparable to KGE only if KGE is
    run under the same sampled protocol.
    """
    from kg_llm.eval.ranking import aggregate

    gen = torch.Generator().manual_seed(seed)
    k = num_candidates - 1
    ranks: list[float] = []

    for h, r, t in tqdm(triples.tolist(), desc="LLM eval", unit="triple"):
        excl = set(filtered_index.true_tails(h, r).tolist())
        excl.add(t)
        cand = torch.tensor([t] + _sample_negatives(num_entities, k, excl, gen), dtype=torch.long)
        ranks.append(_realistic_rank(scorer.score_tail_candidates(h, r, cand)))

        excl = set(filtered_index.true_heads(r, t).tolist())
        excl.add(h)
        cand = torch.tensor([h] + _sample_negatives(num_entities, k, excl, gen), dtype=torch.long)
        ranks.append(_realistic_rank(scorer.score_head_candidates(r, t, cand)))

    return aggregate(torch.tensor(ranks, dtype=torch.float), ks)
