"""Adapter: a trained PyKEEN model exposed through our `Scorer` protocol.

Because we trained on OUR ID space (see scripts/train_kge.py), the scores PyKEEN
returns for "all entities" are already indexed exactly like our data loader and
filtered index — so this adapter is a thin shim, no ID translation needed.

PyKEEN gives us two batch scoring calls:
  - `model.score_t(hr_batch)`: hr_batch is (batch, 2) of [head_id, relation_id];
    returns (batch, num_entities) scores over all candidate tails.
  - `model.score_h(rt_batch)`: rt_batch is (batch, 2) of [relation_id, tail_id];
    returns (batch, num_entities) scores over all candidate heads.
These line up one-to-one with our Scorer's score_tails / score_heads.
"""

from __future__ import annotations

from pathlib import Path

import torch


class PyKEENScorer:
    """Wrap a trained PyKEEN model so it satisfies `kg_llm.eval.ranking.Scorer`."""

    def __init__(self, model) -> None:
        self.model = model.eval()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        hr = torch.stack([heads, relations], dim=1).to(self.device)
        # Return on CPU: our ranking loop is plain Python and runs on CPU tensors.
        return self.model.score_t(hr).cpu()

    @torch.no_grad()
    def score_heads(self, relations: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        rt = torch.stack([relations, tails], dim=1).to(self.device)
        return self.model.score_h(rt).cpu()


def load_pykeen_scorer(model_dir: str | Path) -> PyKEENScorer:
    """Load a model saved by `pipeline(...).save_to_directory()` and wrap it.

    PyKEEN writes the trained model to ``trained_model.pkl`` inside that dir.
    """
    model_path = Path(model_dir) / "trained_model.pkl"
    model = torch.load(model_path, weights_only=False)
    return PyKEENScorer(model)
