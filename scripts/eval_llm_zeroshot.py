"""LLM KG-completion eval via the HuggingFace stack (transformers + peft).

Offline-friendly (transformers/peft are easy to stage / often preinstalled). Works
for the zero-shot base model and SFT/DPO adapters (--adapter, --chat). Two modes:
  default : sampled ranking (gold vs --num-candidates), fast dev metric.
  --full  : rank gold against ALL entities (filtered) — SOTA-comparable, report grade.

Offline: set HF_HUB_OFFLINE=1 and pass local paths to --model / --adapter.
    python scripts/eval_llm_zeroshot.py --model /kaggle/input/qwen-3/.../1.7b \
        --adapter /kaggle/input/<adapter>/qwen3_1.7b_sft_fb15k237 --chat --full --num-test 2000
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.eval.ranking import evaluate
from kg_llm.llm.scorer import LLMScorer, evaluate_llm_sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--num-test", type=int, default=1000, help="random test-triple subset size")
    ap.add_argument("--num-candidates", type=int, default=256, help="sampled mode only")
    ap.add_argument("--full", action="store_true", help="rank vs ALL entities (report-grade)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cand-batch-size", type=int, default=256)
    ap.add_argument("--no-length-norm", action="store_true")
    ap.add_argument("--adapter", default=None, help="path to a LoRA adapter dir (SFT/DPO eval)")
    ap.add_argument("--chat", action="store_true", help="score via chat template (for SFT/DPO models)")
    args = ap.parse_args()

    if args.adapter:
        import os

        if not os.path.isdir(args.adapter):
            raise SystemExit(
                f"--adapter '{args.adapter}' is not a local directory (it got treated as a "
                "HF repo id). Pass the ABSOLUTE path to the extracted adapter."
            )
        args.adapter = os.path.abspath(args.adapter)

    ds = load_fb15k237(args.data_dir)
    filtered_index = ds.build_filtered_index()

    print(f"Loading {args.model} ...")
    print(f"  CUDA available: {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    if args.adapter:
        from peft import PeftModel

        print(f"  applying LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()  # fold adapter into the base for fast inference
    print(f"  model device: {next(model.parameters()).device}  dtype: {dtype}")

    scorer = LLMScorer(
        model, tok, ds,
        length_normalize=not args.no_length_norm,
        cand_batch_size=args.cand_batch_size,
        chat_template=args.chat,
    )

    n = min(args.num_test, ds.test_triples.shape[0])
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(ds.test_triples.shape[0], generator=gen)[:n]
    subset = ds.test_triples[idx]

    if args.full:
        print(f"FULL-candidate eval on {n} test triples (all {ds.num_entities} entities) ...")
        metrics = evaluate(scorer, subset, filtered_index, batch_size=1)
        mode = "FULL-candidate filtered"
    else:
        print(f"Sampled eval on {n} triples, gold vs {args.num_candidates - 1} negatives ...")
        metrics = evaluate_llm_sampled(
            scorer, subset, filtered_index, ds.num_entities,
            num_candidates=args.num_candidates, seed=args.seed,
        )
        mode = f"{args.num_candidates}-way sampled"

    tag = f"{args.model} + adapter" if args.adapter else f"{args.model} (zero-shot)"
    print(f"\n{tag} — FB15k-237 (n={n}, {mode}, filtered):")
    print(f"  {metrics}")


if __name__ == "__main__":
    main()
