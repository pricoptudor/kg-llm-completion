"""Validate LLMScorer's log-prob math with a fake model + tokenizer (no GPU/LLM).

The fake model returns the SAME logits at every position (a fixed preference
vector), so an entity name's score is just the sum/mean of the fixed log-probs of
its tokens — computable by hand. We assert both the KV-cache path and the
re-encode fallback path produce those exact values.
"""

import pytest
import torch
import torch.nn as nn

from kg_llm.llm.scorer import LLMScorer


class _Out:
    def __init__(self, logits, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values


class FakeModel(nn.Module):
    """Constant preference vector as logits at every position; fabricates a cache."""

    def __init__(self, pref: torch.Tensor):
        super().__init__()
        self.register_buffer("pref", pref)
        self._p = nn.Parameter(torch.zeros(1))  # so next(model.parameters()) works

    def forward(self, input_ids, attention_mask=None, past_key_values=None,
                position_ids=None, use_cache=False):
        b, length = input_ids.shape
        logits = self.pref.view(1, 1, -1).expand(b, length, -1).clone()
        pkv = (((torch.zeros(b, 1, length, 1), torch.zeros(b, 1, length, 1)),)) if use_cache else None
        return _Out(logits, pkv)

    def eval(self):
        return self


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class FakeTokenizer:
    """Whitespace tokenizer with an auto-growing, deterministic vocab."""

    def __init__(self):
        self.vocab = {}
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self._next = 1  # reserve 0 for pad

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = self._next
            self._next += 1
        return self.vocab[w]

    def __call__(self, text, add_special_tokens=True):
        return _Enc([self._id(w) for w in text.split()])


class _DS:
    num_entities = 3
    _names = {0: "alpha", 1: "beta", 2: "delta epsilon"}  # entity 2 has a 2-token name

    def entity_name(self, i):
        return self._names[i]

    def relation_name(self, i):
        return "rel"


def _make(pref_vals, use_cache, **kw):
    tok = FakeTokenizer()
    ds = _DS()
    cand = [tok(" " + ds.entity_name(i), add_special_tokens=False).input_ids for i in range(3)]
    ids = {"a": cand[0][0], "b": cand[1][0], "d": cand[2][0], "e": cand[2][1]}
    V = tok._next + 8
    pref = torch.zeros(V)
    for key, val in pref_vals.items():
        pref[ids[key]] = val
    scorer = LLMScorer(FakeModel(pref), tok, ds, use_prefix_cache=use_cache,
                       cand_batch_size=2, **kw)
    return scorer, torch.log_softmax(pref, dim=-1), ids


@pytest.mark.parametrize("use_cache", [True, False])
def test_logprob_and_length_normalization(use_cache):
    scorer, lp, ids = _make({"a": 2.0, "b": 1.0, "d": 0.5, "e": 0.5}, use_cache)
    scores = scorer.score_tails(torch.tensor([0]), torch.tensor([0]))[0]
    assert torch.allclose(scores[0], lp[ids["a"]], atol=1e-5)         # 1-token
    assert torch.allclose(scores[1], lp[ids["b"]], atol=1e-5)         # 1-token
    assert torch.allclose(scores[2], (lp[ids["d"]] + lp[ids["e"]]) / 2, atol=1e-5)  # mean of 2
    assert scores[0] > scores[1] > scores[2]


@pytest.mark.parametrize("use_cache", [True, False])
def test_sum_not_mean_when_normalization_off(use_cache):
    scorer, lp, ids = _make({"d": 0.5, "e": 0.5}, use_cache, length_normalize=False)
    scores = scorer.score_tails(torch.tensor([0]), torch.tensor([0]))[0]
    assert torch.allclose(scores[2], lp[ids["d"]] + lp[ids["e"]], atol=1e-5)  # raw sum


def test_cached_and_fallback_agree():
    """The KV-cache path and the re-encode fallback must give identical scores."""
    cached, _, _ = _make({"a": 2.0, "b": 1.0, "d": 0.5, "e": 0.5}, True)
    fallback, _, _ = _make({"a": 2.0, "b": 1.0, "d": 0.5, "e": 0.5}, False)
    s1 = cached.score_tails(torch.tensor([0]), torch.tensor([0]))[0]
    s2 = fallback.score_tails(torch.tensor([0]), torch.tensor([0]))[0]
    assert torch.allclose(s1, s2, atol=1e-5)
