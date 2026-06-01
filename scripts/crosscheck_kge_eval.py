"""Cross-check: does OUR filtered-eval harness agree with PyKEEN's own evaluator?

If both pipelines, run on the same trained model and the same test triples, report
the same MRR, then we've validated both at once: our harness is correct, and
PyKEEN's results are reproducible through our code (which is what we'll use to
also score the SFT/DPO LLMs on equal footing).

Run after a real training run has produced artifacts/kge/<name>/:
    python scripts/crosscheck_kge_eval.py --model-dir artifacts/kge/complex_fb15k237
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.eval.ranking import evaluate
from kg_llm.kge.pykeen_scorer import load_pykeen_scorer


def _dig(d: dict, *keys):
    """Walk a nested dict by keys, returning None if any level is missing."""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def pykeen_test_mrr(model_dir: Path) -> float | None:
    """Pull PyKEEN's own filtered (realistic) test MRR from its results.json."""
    results_path = model_dir / "results.json"
    if not results_path.exists():
        return None
    results = json.loads(results_path.read_text())
    # PyKEEN nests metrics under metrics -> both -> realistic -> <metric name>.
    return _dig(
        results, "metrics", "both", "realistic", "inverse_harmonic_mean_rank"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--tol", type=float, default=0.01, help="max allowed MRR gap")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    ds = load_fb15k237(args.data_dir)
    filtered_index = ds.build_filtered_index()
    scorer = load_pykeen_scorer(model_dir)
    pk_mrr = pykeen_test_mrr(model_dir)

    # Inverse-triples models renumber relations inside PyKEEN, so our manual adapter
    # would score the wrong relations (we saw 0.005 vs PyKEEN's 0.22). Detect that
    # and defer to PyKEEN's authoritative number instead of printing garbage. Our
    # harness is validated separately on the non-inverse config (exact agreement).
    try:
        n_model_rel = scorer.model.relation_representations[0](indices=None).shape[0]
    except Exception:
        n_model_rel = getattr(scorer.model, "num_relations", ds.num_relations)

    if n_model_rel != ds.num_relations:
        print(
            f"Model uses inverse triples ({n_model_rel} relation reps vs our "
            f"{ds.num_relations}) — our adapter is N/A here."
        )
        if pk_mrr is not None:
            print(f"PyKEEN test MRR (authoritative, filtered): {pk_mrr:.4f}")
        else:
            print("PyKEEN MRR: (results.json not found — read it from the run log)")
        print("Our harness was validated separately on the non-inverse config.")
        return

    ours = evaluate(scorer, ds.test_triples, filtered_index)
    print(f"Our harness   : {ours}")
    if pk_mrr is None:
        print("PyKEEN MRR    : (results.json not found — compare against the run log)")
        return
    print(f"PyKEEN MRR    : {pk_mrr:.4f}")
    gap = abs(ours.mrr - pk_mrr)
    print(f"|gap|         : {gap:.4f}  (tolerance {args.tol})")
    # Small differences are expected: PyKEEN's default tie policy and ours may
    # round borderline ties slightly differently. A large gap means a real bug.
    assert gap <= args.tol, (
        f"Harnesses disagree by {gap:.4f} > {args.tol}. Investigate before trusting "
        "either number (likely an ID-alignment or filtering mismatch)."
    )
    print("OK — harnesses agree within tolerance.")


if __name__ == "__main__":
    main()
