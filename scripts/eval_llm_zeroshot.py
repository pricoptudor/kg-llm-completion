"""Zero-shot LLM baseline on FB15k-237 via log-prob scoring.

Runs a base instruct model through the SAME filtered-eval harness as the KGE
baselines, on a fixed random subset of test triples (full-test LLM scoring is
expensive). Expect poor numbers — this is the documented floor SFT/DPO must beat.

Run on a GPU box / Kaggle. Start with a tiny dry run to validate plumbing:
    python scripts/eval_llm_zeroshot.py --num-test 5
Then the real baseline:
    python scripts/eval_llm_zeroshot.py --num-test 1000
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.eval.ranking import evaluate
from kg_llm.llm.scorer import LLMScorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--num-test", type=int, default=1000, help="random test-triple subset size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cand-batch-size", type=int, default=128)
    ap.add_argument("--no-length-norm", action="store_true")
    ap.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="disable KV-cache prefix reuse (slower, max compatibility) if the cache API errors",
    )
    args = ap.parse_args()

    ds = load_fb15k237(args.data_dir)
    filtered_index = ds.build_filtered_index()

    print(f"Loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    scorer = LLMScorer(
        model,
        tok,
        ds,
        length_normalize=not args.no_length_norm,
        cand_batch_size=args.cand_batch_size,
        use_prefix_cache=not args.no_prefix_cache,
    )

    # Fixed random subset of test triples (reproducible via seed). The SAME subset
    # should be used for any KGE comparison so the numbers are apples-to-apples.
    n = min(args.num_test, ds.test_triples.shape[0])
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(ds.test_triples.shape[0], generator=gen)[:n]
    subset = ds.test_triples[idx]

    print(f"Scoring {n} test triples (= {2 * n} head+tail queries) over "
          f"{ds.num_entities} candidates each ...")
    metrics = evaluate(scorer, subset, filtered_index, batch_size=1)

    print(f"\nZero-shot {args.model} — FB15k-237 test subset (n={n}, filtered):")
    print(f"  {metrics}")
    print("\nThis is expected to be low — it is the floor SFT/DPO must beat.")
    print("For a fair table later, evaluate the KGE models on this same subset/seed.")


if __name__ == "__main__":
    main()
