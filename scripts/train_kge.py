"""Train a KGE baseline on FB15k-237 with PyKEEN, on OUR entity/relation IDs.

The one non-obvious thing this script does is **ID alignment**. PyKEEN normally
assigns its own integer IDs to entities and relations. If we let it, the
embeddings it produces would be indexed differently from our data loader, and
nothing else in the project (our filtered-eval harness, the Week 3 hard-negative
miner) would line up with them. So we build PyKEEN's TriplesFactory with OUR
`entity_to_id` / `relation_to_id` maps — one ID space for the whole project.

Run on a GPU box / Kaggle (training is not for the laptop):
    python scripts/train_kge.py --config configs/kge/complex.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from kg_llm.data.fb15k237 import load_fb15k237


def _labeled(triples: torch.Tensor, ds) -> np.ndarray:
    """Turn our (N,3) id tensor back into the (N,3) array of *string* labels
    (MID / relation-path / MID) that PyKEEN's TriplesFactory consumes."""
    arr = triples.numpy()
    out = np.empty((arr.shape[0], 3), dtype=object)
    out[:, 0] = [ds.id_to_entity[i] for i in arr[:, 0]]
    out[:, 1] = [ds.id_to_relation[i] for i in arr[:, 1]]
    out[:, 2] = [ds.id_to_entity[i] for i in arr[:, 2]]
    return out


def build_factories(ds):
    """Three TriplesFactories (train/valid/test) that all share OUR ID space."""
    common = dict(entity_to_id=ds.entity_to_id, relation_to_id=ds.relation_to_id)
    train = TriplesFactory.from_labeled_triples(_labeled(ds.train_triples, ds), **common)
    valid = TriplesFactory.from_labeled_triples(_labeled(ds.valid_triples, ds), **common)
    test = TriplesFactory.from_labeled_triples(_labeled(ds.test_triples, ds), **common)
    return train, valid, test


def _pipeline_kwargs(cfg: dict) -> dict:
    """Map our YAML config onto PyKEEN's pipeline() arguments, passing only the
    keys that are actually present so unset knobs fall back to PyKEEN defaults."""
    passthrough = [
        "model_kwargs",
        "loss",
        "loss_kwargs",
        "optimizer",
        "optimizer_kwargs",
        "regularizer",
        "regularizer_kwargs",
        "training_loop",
        "training_kwargs",
        "negative_sampler",
        "negative_sampler_kwargs",
        "stopper",
        "stopper_kwargs",
    ]
    kwargs = {k: cfg[k] for k in passthrough if k in cfg}
    kwargs["model"] = cfg["model"]
    kwargs["random_seed"] = cfg.get("random_seed", 42)
    if cfg.get("wandb"):
        kwargs["result_tracker"] = "wandb"
        kwargs["result_tracker_kwargs"] = cfg["wandb"]
    return kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--output-dir", default="artifacts/kge")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ds = load_fb15k237(args.data_dir)
    train, valid, test = build_factories(ds)
    print(
        f"Aligned TriplesFactories on our IDs: "
        f"{train.num_entities} entities, {train.num_relations} relations."
    )

    result = pipeline(
        training=train,
        validation=valid,
        testing=test,
        **_pipeline_kwargs(cfg),
    )

    out = Path(args.output_dir) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    # PyKEEN's own bundle: trained model, metrics, config, training curve.
    result.save_to_directory(str(out))

    # Also dump raw embeddings indexed by OUR ids, with the maps alongside, so the
    # Week 3 hard-negative miner can load them without depending on PyKEEN.
    model = result.model
    entity_emb = model.entity_representations[0](indices=None).detach().cpu()
    relation_emb = model.relation_representations[0](indices=None).detach().cpu()
    torch.save(
        {
            "entity_embeddings": entity_emb,
            "relation_embeddings": relation_emb,
            "entity_to_id": ds.entity_to_id,
            "relation_to_id": ds.relation_to_id,
            "model": cfg["model"],
        },
        out / "embeddings.pt",
    )

    mrr = result.metric_results.get_metric("both.realistic.inverse_harmonic_mean_rank")
    h1 = result.metric_results.get_metric("both.realistic.hits_at_1")
    h10 = result.metric_results.get_metric("both.realistic.hits_at_10")
    print(f"\n{cfg['name']} — PyKEEN test (filtered, realistic):")
    print(f"  MRR={mrr:.4f}  H@1={h1:.4f}  H@10={h10:.4f}")
    print(f"  saved model + embeddings to {out}")


if __name__ == "__main__":
    main()
