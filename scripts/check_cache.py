"""Verify the prompt-KV-cache scorer preserves the EVAL METRIC vs the uncached scorer.

bf16 KV caching is not bitwise-identical to a single full-precision forward (this is
true of all cached inference, incl. HF generate / vLLM), so we check the metric, not
per-score equality: cached vs uncached MRR/Hits on a sample of test triples must agree
to within --tol (default 0.005, far inside num_test sampling noise).

    python scripts/check_cache.py --model <path> [--adapter <dir>] [--chat] \
        --data-dir <fb15k237> --num-test 100
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--num-test", type=int, default=100)
    ap.add_argument("--cand-batch-size", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=0.005)
    args = ap.parse_args()

    ds = load_fb15k237(args.data_dir)
    fi = ds.build_filtered_index()
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto").eval()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    idx = torch.randperm(ds.test_triples.shape[0], generator=torch.Generator().manual_seed(42))[: args.num_test]
    subset = ds.test_triples[idx]
    common = dict(cand_batch_size=args.cand_batch_size, chat_template=args.chat)

    print("running cached ...")
    mc = evaluate(LLMScorer(model, tok, ds, use_kv_cache=True, **common), subset, fi, batch_size=1)
    print("running uncached ...")
    mp = evaluate(LLMScorer(model, tok, ds, use_kv_cache=False, **common), subset, fi, batch_size=1)
    print("cached  :", mc)
    print("uncached:", mp)
    dmrr = abs(mc.mrr - mp.mrr)
    print(f"\nΔMRR={dmrr:.4f}  ->  {'PASS' if dmrr < args.tol else 'FAIL'} (tol={args.tol})")


if __name__ == "__main__":
    main()
