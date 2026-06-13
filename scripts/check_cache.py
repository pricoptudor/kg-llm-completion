"""Verify the prompt-KV-cache scorer is numerically identical to the uncached one.

For a few test triples, score ALL entities both ways (cached vs uncached) for the
tail and head query, and report the max abs difference. They should agree to <1e-3
(tiny float reordering only). Run this once before trusting the cached fast path.

    python scripts/check_cache.py --model <path> [--adapter <dir>] [--chat] \
        --data-dir <fb15k237> --num-test 3
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.llm.scorer import LLMScorer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--num-test", type=int, default=3)
    ap.add_argument("--cand-batch-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_fb15k237(args.data_dir)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    tok = AutoTokenizer.from_pretrained(args.adapter or args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    common = dict(cand_batch_size=args.cand_batch_size, chat_template=args.chat)
    cached = LLMScorer(model, tok, ds, use_kv_cache=True, **common)
    plain = LLMScorer(model, tok, ds, use_kv_cache=False, **common)

    allidx = torch.arange(ds.num_entities)
    gen = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(ds.test_triples.shape[0], generator=gen)[: args.num_test]
    worst = 0.0
    for h, r, t in ds.test_triples[idx].tolist():
        for name, pre in (("tail", cached._tail_prefix(h, r)), ("head", cached._head_prefix(r, t))):
            sc = cached._score_indices_cached(pre, allidx)
            sp = plain._score_indices_nocache(pre, allidx)
            d = (sc - sp).abs().max().item()
            worst = max(worst, d)
            # also confirm the gold's rank matches under both
            gold = t if name == "tail" else h
            rc = int((sc > sc[gold]).sum()) + 1
            rp = int((sp > sp[gold]).sum()) + 1
            print(f"  {name} (h={h},r={r},t={t}): max|Δ|={d:.2e}  rank cached={rc} uncached={rp}")
    print(f"\nworst max|Δ| over {args.num_test} triples: {worst:.2e}  -> "
          f"{'PASS' if worst < 1e-3 else 'FAIL (do NOT trust cache)'}")


if __name__ == "__main__":
    main()
