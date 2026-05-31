"""Brute-force verification of the filtered ranking harness on a toy KG.

Every expected number here is computed by hand in the comments so the test is a
genuine independent check of the protocol, not a restatement of the code.
"""

import torch

from kg_llm.data.fb15k237 import FilteredIndex
from kg_llm.eval.ranking import aggregate, evaluate, filtered_ranks


def test_filtering_rescues_rank():
    """A known-true distractor scoring above the gold must not cost it rank.

    Entities {0,1,2,3}. Query (h=*, r=*, ?), gold tail = 1.
    Known-true tails for this (h,r) = {1, 2}  ->  entity 2 is a correct answer.
    Scores over [e0,e1,e2,e3] = [0.1, 0.5, 0.9, 0.2].

      RAW: sorted desc -> e2(0.9), e1(0.5), e3(0.2), e0(0.1); gold e1 is rank 2.
      FILTERED: e2 is known-true -> removed. Gold e1=0.5 beats e3,e0 -> rank 1.
    """
    scores = torch.tensor([[0.1, 0.5, 0.9, 0.2]])
    gold = torch.tensor([1])
    filt = [torch.tensor([1, 2])]  # known-true tails (includes the gold)
    ranks = filtered_ranks(scores, gold, filt)
    assert ranks.tolist() == [1.0]


def test_realistic_tie_handling():
    """Ties split half-and-half (optimistic+pessimistic)/2.

    Scores [0.5, 0.5, 0.2], gold = 0, only the gold is known-true.
    Gold scores 0.5; entity 1 ties at 0.5 (a genuine negative).
      strictly greater = 0, tied (excl. gold) = 1  ->  rank = 1 + 0 + 0.5 = 1.5.
    """
    scores = torch.tensor([[0.5, 0.5, 0.2]])
    gold = torch.tensor([0])
    filt = [torch.tensor([0])]
    ranks = filtered_ranks(scores, gold, filt)
    assert ranks.tolist() == [1.5]


def test_no_filter_plain_rank():
    """With no extra true answers, filtered rank == raw rank.

    Scores [0.9, 0.5, 0.2], gold = 1.  One entity (e0=0.9) beats it -> rank 2.
    """
    scores = torch.tensor([[0.9, 0.5, 0.2]])
    gold = torch.tensor([1])
    filt = [torch.tensor([1])]  # only the gold is known-true
    ranks = filtered_ranks(scores, gold, filt)
    assert ranks.tolist() == [2.0]


def test_aggregate_metrics():
    """MRR and Hits@k from a hand-picked set of ranks.

    ranks = [1, 2, 4, 1.5]
      MRR    = mean(1, 0.5, 0.25, 0.6667) = 2.41667/4 = 0.604166...
      Hits@1 = mean(rank<=1)  = [T,F,F,F] = 0.25
      Hits@3 = mean(rank<=3)  = [T,T,F,T] = 0.75
      Hits@10= all true       = 1.0
    """
    ranks = torch.tensor([1.0, 2.0, 4.0, 1.5])
    m = aggregate(ranks)
    assert abs(m.mrr - 0.6041666) < 1e-5
    assert abs(m.hits_at_1 - 0.25) < 1e-9
    assert abs(m.hits_at_3 - 0.75) < 1e-9
    assert abs(m.hits_at_10 - 1.0) < 1e-9
    assert m.num_queries == 4


class _ToyScorer:
    """Deterministic scorer over a 3-entity KG, defined by fixed score matrices
    keyed on the (entity, relation) ids it is asked about."""

    def __init__(self, tail_table, head_table):
        self._tail = tail_table  # dict[(h, r)] -> list[float] over entities
        self._head = head_table  # dict[(r, t)] -> list[float] over entities

    def score_tails(self, heads, relations):
        return torch.tensor(
            [self._tail[(int(h), int(r))] for h, r in zip(heads, relations)]
        )

    def score_heads(self, relations, tails):
        return torch.tensor(
            [self._head[(int(r), int(t))] for r, t in zip(relations, tails)]
        )


def test_evaluate_pools_head_and_tail():
    """End-to-end: one test triple -> two queries, pooled.

    Entities {0,1,2}, relation {0}. Known-true triples = {(0,0,1), (0,0,2)}.
    Test triple = (0, 0, 1).

    Tail query (0,0,?), gold=1: scores [0.2, 0.9, 0.5].
      true tails {1,2} -> mask e2. Gold e1=0.9 is top -> rank 1.
    Head query (?,0,1), gold=0: scores [0.7, 0.1, 0.3].
      true heads for (r=0,t=1) = {0} -> mask none extra. Gold e0=0.7 top -> rank 1.

    Pooled ranks = [1, 1] -> MRR 1.0, all hits 1.0, num_queries = 2.
    """
    triples = torch.tensor([[0, 0, 1]])
    fi = FilteredIndex(
        hr_to_tails={(0, 0): torch.tensor([1, 2])},
        rt_to_heads={(0, 1): torch.tensor([0]), (0, 2): torch.tensor([0])},
    )
    scorer = _ToyScorer(
        tail_table={(0, 0): [0.2, 0.9, 0.5]},
        head_table={(0, 1): [0.7, 0.1, 0.3]},
    )
    m = evaluate(scorer, triples, fi)
    assert m.num_queries == 2
    assert abs(m.mrr - 1.0) < 1e-9
    assert abs(m.hits_at_1 - 1.0) < 1e-9
