"""Filtered ranking metrics for knowledge graph completion.

Implements the standard *filtered* evaluation protocol (Bordes et al. 2013,
"Translating Embeddings for Modeling Multi-relational Data") with **realistic**
tie handling — the optimistic/pessimistic average, which is PyKEEN's default and
the one that keeps a degenerate constant-scoring model from looking strong.

The math (full walk-through lives in the project chat notes):

  For a test triple (h, r, t) we form two queries — predict the tail (h, r, ?)
  and predict the head (?, r, t). For the tail query we score every candidate
  entity e, giving s(h, r, e); higher = more plausible. The rank of the gold t
  is "how many candidates beat it, plus one":

      rank(t) = 1 + #{ e : s(h,r,e) > s(h,r,t) }

  *Filtered*: before counting, we drop every OTHER known-true entity t' with
  (h, r, t') in train+valid+test, so the gold competes only against genuine
  negatives. *Realistic ties*: candidates with exactly the gold's score
  contribute half each:

      rank(t) = 1 + (#strictly greater) + 0.5 * (#tied, excluding the gold)

  Metrics, pooled over all head and tail queries:

      MRR     = mean(1 / rank)
      Hits@k  = mean(rank <= k)            (k in {1, 3, 10})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from kg_llm.data.fb15k237 import FB15k237, FilteredIndex


class Scorer(Protocol):
    """Anything that can score candidate entities for KG-completion queries.

    Both methods take equal-length 1-D LongTensors and return a
    ``(batch, num_entities)`` float tensor where row ``i`` holds the plausibility
    of every entity as the missing slot for query ``i`` (higher = more plausible).

    This is the single interface every model plugs into — the trivial frequency
    baseline (Task 6), the PyKEEN KGE models, and the LLM log-prob scorer all
    implement it, so the eval harness never needs to know which is which.
    """

    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Score candidate tails for (h, r, ?). Returns (batch, num_entities)."""
        ...

    def score_heads(self, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        """Score candidate heads for (?, r, t). Returns (batch, num_entities)."""
        ...


@dataclass
class RankingMetrics:
    """Aggregated filtered metrics over a pooled set of queries."""

    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float
    num_queries: int

    def __str__(self) -> str:
        return (
            f"MRR={self.mrr:.4f}  H@1={self.hits_at_1:.4f}  "
            f"H@3={self.hits_at_3:.4f}  H@10={self.hits_at_10:.4f}  "
            f"(n={self.num_queries})"
        )


def filtered_ranks(
    scores: torch.Tensor,
    gold_ids: torch.Tensor,
    filter_sets: list[torch.Tensor],
) -> torch.Tensor:
    """Compute realistic filtered ranks for a batch of queries.

    Args:
        scores: (batch, num_entities) float plausibility scores.
        gold_ids: (batch,) long, the held-out correct entity per query.
        filter_sets: length-``batch`` list; ``filter_sets[i]`` is a 1-D long
            tensor of all known-true entity ids for query ``i``. It *may* include
            the gold itself (it usually does, since (h, r, t) is a true triple) —
            we handle that by re-exposing the gold after masking.

    Returns:
        (batch,) float tensor of realistic ranks (1.0 = perfect).
    """
    batch = scores.shape[0]
    ranks = torch.empty(batch, dtype=torch.float)
    neg_inf = float("-inf")

    for i in range(batch):
        row = scores[i].clone()
        gold = int(gold_ids[i])
        gold_score = row[gold].item()

        # Remove all known-true entities from contention: they are correct
        # answers, not distractors, so they must not push the gold's rank down.
        filt = filter_sets[i]
        if filt.numel() > 0:
            row[filt] = neg_inf
        # ...but the gold must stay in the running — restore just its score.
        row[gold] = gold_score

        greater = int((row > gold_score).sum())
        # Entities tied with the gold, excluding the gold itself.
        tied = int((row == gold_score).sum()) - 1
        ranks[i] = 1.0 + greater + 0.5 * tied

    return ranks


def aggregate(ranks: torch.Tensor, ks: tuple[int, ...] = (1, 3, 10)) -> RankingMetrics:
    """Reduce a 1-D tensor of ranks to MRR and Hits@k."""
    mrr = (1.0 / ranks).mean().item()
    hits = {k: (ranks <= k).float().mean().item() for k in ks}
    return RankingMetrics(
        mrr=mrr,
        hits_at_1=hits[1],
        hits_at_3=hits[3],
        hits_at_10=hits[10],
        num_queries=int(ranks.numel()),
    )


@torch.no_grad()
def evaluate(
    scorer: Scorer,
    triples: torch.Tensor,
    filtered_index: FilteredIndex,
    ks: tuple[int, ...] = (1, 3, 10),
    batch_size: int = 128,
) -> RankingMetrics:
    """Run filtered evaluation over ``triples``, pooling head and tail queries.

    Each triple contributes two queries (predict tail, predict head), so the
    returned ``num_queries`` is ``2 * len(triples)`` — the standard FB15k-237
    protocol. Comparing only one direction breaks comparability with published
    numbers.
    """
    all_ranks: list[torch.Tensor] = []

    for start in range(0, triples.shape[0], batch_size):
        batch = triples[start : start + batch_size]
        h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

        # Tail prediction: (h, r, ?), gold = t, filter other true tails.
        tail_scores = scorer.score_tails(h, r)
        tail_filters = [
            filtered_index.true_tails(int(h[i]), int(r[i])) for i in range(batch.shape[0])
        ]
        all_ranks.append(filtered_ranks(tail_scores, t, tail_filters))

        # Head prediction: (?, r, t), gold = h, filter other true heads.
        head_scores = scorer.score_heads(r, t)
        head_filters = [
            filtered_index.true_heads(int(r[i]), int(t[i])) for i in range(batch.shape[0])
        ]
        all_ranks.append(filtered_ranks(head_scores, h, head_filters))

    return aggregate(torch.cat(all_ranks), ks)
