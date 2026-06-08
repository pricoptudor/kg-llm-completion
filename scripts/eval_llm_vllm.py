"""LLM KG-completion eval via vLLM (fast log-prob scoring).

Same sampled protocol as scripts/eval_llm_zeroshot.py, but backed by vLLM for
speed. Works for the zero-shot base model and for SFT/DPO adapters (--adapter,
--chat). Extends to full-candidate (report-grade) eval by raising --num-candidates.

Validate first against the HF result on the 1.7B SFT model (expect ~0.418):
    python scripts/eval_llm_vllm.py --model Qwen/Qwen3-1.7B \
        --adapter /abs/path/qwen3_1.7b_sft_fb15k237 --chat --num-test 50
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer
from vllm import LLM
from vllm.lora.request import LoRARequest

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.llm.scorer import evaluate_llm_sampled
from kg_llm.llm.vllm_scorer import VLLMScorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None, help="abs path to a LoRA adapter dir")
    ap.add_argument("--chat", action="store_true", help="chat-template scoring (SFT/DPO models)")
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--num-test", type=int, default=1000)
    ap.add_argument("--num-candidates", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=256)
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
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.9,
    )
    lora_request = None
    if args.adapter:
        llm_kwargs.update(enable_lora=True, max_lora_rank=16)
    llm = LLM(**llm_kwargs)
    if args.adapter:
        lora_request = LoRARequest("adapter", 1, args.adapter)

    scorer = VLLMScorer(
        llm, tok, ds, chat_template=args.chat, lora_request=lora_request
    )

    n = min(args.num_test, ds.test_triples.shape[0])
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(ds.test_triples.shape[0], generator=gen)[:n]
    subset = ds.test_triples[idx]

    print(f"vLLM scoring {n} test triples, gold vs {args.num_candidates - 1} negatives ...")
    metrics = evaluate_llm_sampled(
        scorer, subset, filtered_index, ds.num_entities,
        num_candidates=args.num_candidates, seed=args.seed,
    )
    tag = f"{args.model} + adapter" if args.adapter else f"{args.model} (zero-shot)"
    print(f"\n[vLLM] {tag} — FB15k-237 (n={n}, {args.num_candidates}-way sampled, filtered):")
    print(f"  {metrics}")


if __name__ == "__main__":
    main()
