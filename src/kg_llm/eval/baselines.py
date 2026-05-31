"""Trivial baselines for sanity-checking the evaluation harness.

These are not meant to be competitive — they exist to prove the ranking harness
produces sensible numbers end-to-end before we trust it on real KGE/LLM scores,
and to provide an honest floor in the results table.
"""

from __future__ import annotations

import torch


class FrequencyBaseline:
    """Relation-conditional frequency scorer.

    Scores a candidate entity ``e`` for query ``(h, r, ?)`` by how often ``e``
    appeared as the tail of relation ``r`` in the training triples — ignoring
    ``h`` entirely. Symmetric for head queries. This is the classic
    "most-frequent-object-for-this-relation" floor: it knows nothing about the
    specific subject, only the marginal popularity of objects under each relation.

    Frequencies are counted on TRAIN ONLY; counting valid/test would leak the
    held-out answers into the scores.
    """

    def __init__(self, tail_freq: torch.Tensor, head_freq: torch.Tensor) -> None:
        # Both are (num_relations, num_entities) float tensors.
        self.tail_freq = tail_freq
        self.head_freq = head_freq

    @classmethod
    def fit(
        cls, train_triples: torch.Tensor, num_entities: int, num_relations: int
    ) -> "FrequencyBaseline":
        """Accumulate per-relation head/tail counts from the training triples."""
        tail_freq = torch.zeros(num_relations, num_entities)
        head_freq = torch.zeros(num_relations, num_entities)
        h, r, t = train_triples[:, 0], train_triples[:, 1], train_triples[:, 2]
        ones = torch.ones(train_triples.shape[0])
        # index_put_ with accumulate=True is a vectorized scatter-add: for every
        # training triple, +1 at [relation, tail] (and [relation, head]).
        tail_freq.index_put_((r, t), ones, accumulate=True)
        head_freq.index_put_((r, h), ones, accumulate=True)
        return cls(tail_freq=tail_freq, head_freq=head_freq)

    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        # (batch, num_entities): the tail-frequency row for each query's relation.
        return self.tail_freq[relations]

    def score_heads(self, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        return self.head_freq[relations]
