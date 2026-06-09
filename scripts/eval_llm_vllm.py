"""LLM KG-completion eval via vLLM (fast log-prob scoring).

Two modes:
  default  : sampled ranking (gold vs --num-candidates), fast dev metric.
  --full   : rank gold against ALL entities (filtered) — SOTA-comparable, report
             grade. Use on a query subset (--num-test) on a capable GPU.

Works for the zero-shot base model and SFT/DPO adapters (--adapter, --chat).
Needs a GPU with compute capability >= 8.0 (Ada/Hopper/Blackwell) so vLLM uses
FlashAttention-2 (no FlashInfer JIT). On an older T4 this path is not supported.

    # dev cross-check (should match the HF 0.418 on the 1.7B SFT model):
    python scripts/eval_llm_vllm.py --model Qwen/Qwen3-1.7B --adapter /abs/path --chat --num-test 50
    # report-grade full-candidate:
    python scripts/eval_llm_vllm.py --model Qwen/Qwen3-1.7B --adapter /abs/path --chat --full --num-test 3000
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm.lora.request import LoRARequest

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.eval.ranking import evaluate
from kg_llm.llm.scorer import evaluate_llm_sampled
from kg_llm.llm.vllm_scorer import VLLMScorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None, help="abs path to a LoRA adapter dir")
    ap.add_argument("--chat", action="store_true", help="chat-template scoring (SFT/DPO models)")
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--num-test", type=int, default=1000)
    ap.add_argument("--num-candidates", type=int, default=256, help="sampled mode only")
    ap.add_argument("--full", action="store_true", help="rank vs ALL entities (report-grade)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=256)
    ap.add_argument("--eager", action="store_true", help="force eager (skip vLLM compile/cudagraph)")
    args = ap.parse_args()

    if args.adapter:
        if not os.path.isdir(args.adapter):
            raise SystemExit(f"--adapter '{args.adapter}' is not a local directory.")
        args.adapter = os.path.abspath(args.adapter)

    ds = load_fb15k237(args.data_dir)
    filtered_index = ds.build_filtered_index()
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)

    llm_kwargs = dict(
        model=args.model,
        dtype="auto",  # use the model's native dtype (bf16 for Qwen3) on a capable GPU
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
        enforce_eager=args.eager,
    )
    lora_request = None
    if args.adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=16)
    llm = LLM(**llm_kwargs)
    if args.adapter:
        lora_request = LoRARequest("adapter", 1, args.adapter)

    scorer = VLLMScorer(llm, tok, ds, chat_template=args.chat, lora_request=lora_request)

    n = min(args.num_test, ds.test_triples.shape[0])
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(ds.test_triples.shape[0], generator=gen)[:n]
    subset = ds.test_triples[idx]

    if args.full:
        print(f"vLLM FULL-candidate eval on {n} test triples (all {ds.num_entities} entities) ...")
        metrics = evaluate(scorer, subset, filtered_index, batch_size=1)
        mode = "FULL-candidate filtered"
    else:
        print(f"vLLM sampled eval on {n} triples, gold vs {args.num_candidates - 1} negatives ...")
        metrics = evaluate_llm_sampled(
            scorer, subset, filtered_index, ds.num_entities,
            num_candidates=args.num_candidates, seed=args.seed,
        )
        mode = f"{args.num_candidates}-way sampled"

    tag = f"{args.model} + adapter" if args.adapter else f"{args.model} (zero-shot)"
    print(f"\n[vLLM] {tag} — FB15k-237 (n={n}, {mode}, filtered):")
    print(f"  {metrics}")


if __name__ == "__main__":
    main()
