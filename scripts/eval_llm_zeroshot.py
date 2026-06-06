"""Zero-shot LLM baseline on FB15k-237 via log-prob scoring.

Ranks the gold answer against a sampled candidate set (full 14,541-way scoring is
infeasible on a T4 for many queries). Expect poor numbers — this is the documented
floor SFT/DPO must beat.

Run on a GPU box / Kaggle. Tiny dry run first to validate + time:
    python scripts/eval_llm_zeroshot.py --num-test 5
Then the real baseline (sampled ranking is cheap, so 1000 is fine):
    python scripts/eval_llm_zeroshot.py --num-test 1000
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.llm.scorer import LLMScorer, evaluate_llm_sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--num-test", type=int, default=1000, help="random test-triple subset size")
    ap.add_argument("--num-candidates", type=int, default=256,
                    help="rank gold against this many candidates (gold + sampled negatives)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cand-batch-size", type=int, default=128)
    ap.add_argument("--no-length-norm", action="store_true")
    args = ap.parse_args()

    ds = load_fb15k237(args.data_dir)
    filtered_index = ds.build_filtered_index()

    print(f"Loading {args.model} ...")
    print(f"  CUDA available: {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"  model device: {next(model.parameters()).device}")

    scorer = LLMScorer(
        model, tok, ds,
        length_normalize=not args.no_length_norm,
        cand_batch_size=args.cand_batch_size,
    )

    n = min(args.num_test, ds.test_triples.shape[0])
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(ds.test_triples.shape[0], generator=gen)[:n]
    subset = ds.test_triples[idx]

    print(f"Scoring {n} test triples (= {2 * n} head+tail queries), "
          f"gold vs {args.num_candidates - 1} sampled negatives each ...")
    metrics = evaluate_llm_sampled(
        scorer, subset, filtered_index, ds.num_entities,
        num_candidates=args.num_candidates, seed=args.seed,
    )

    print(f"\nZero-shot {args.model} — FB15k-237 (n={n} triples, "
          f"{args.num_candidates}-way sampled, filtered):")
    print(f"  {metrics}")
    print("\nLow is expected — the zero-shot floor. NOTE: sampled metric, not "
          "directly comparable to KGE's full-14,541 MRR (run KGE under the same "
          "sampling for a fair head-to-head).")


if __name__ == "__main__":
    main()
