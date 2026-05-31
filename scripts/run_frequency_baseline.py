"""End-to-end smoke test of the Day 1-2 setup.

Loads FB15k-237, fits the relation-conditional frequency baseline on train, and
runs the filtered evaluation harness on the test split. The point is not the
score itself but proving the data loader + filtered index + ranking harness
compose into sane numbers before any real model is involved.

Run from the repo root:
    python scripts/run_frequency_baseline.py
"""

from __future__ import annotations

from pathlib import Path

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.eval.baselines import FrequencyBaseline
from kg_llm.eval.ranking import evaluate

DATA_DIR = Path("data_cache/fb15k237")


def main() -> None:
    ds = load_fb15k237(DATA_DIR)
    print(
        f"Loaded FB15k-237: {ds.num_entities} entities, {ds.num_relations} relations, "
        f"{ds.train_triples.shape[0]} train / {ds.test_triples.shape[0]} test triples."
    )

    filtered_index = ds.build_filtered_index()
    model = FrequencyBaseline.fit(ds.train_triples, ds.num_entities, ds.num_relations)

    metrics = evaluate(model, ds.test_triples, filtered_index)
    print("\nRelation-conditional frequency baseline — FB15k-237 test (filtered):")
    print(f"  {metrics}")
    print(
        "\nSanity expectation: well below KGE (ComplEx ~0.32 MRR) but clearly "
        "above random (~1/14541 ~= 7e-5 MRR)."
    )


if __name__ == "__main__":
    main()
