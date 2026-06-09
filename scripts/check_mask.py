"""Sanity-check answer-only masking before a full SFT run.

Prints, for a few examples, the SUPERVISED tokens (labels != -100 -> loss is
computed on these) vs the MASKED tokens (labels == -100). SUPERVISED should be
exactly the gold entity name + <|im_end|>; MASKED should be the question/prompt.

    python scripts/check_mask.py --model <local model path> --data-dir <fb15k237>
"""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.llm.sft_data import make_sft_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = make_sft_dataset(load_fb15k237(args.data_dir), "train", max_triples=args.n)

    for i in range(min(args.n, len(ds))):
        msgs = ds[i]["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        prompt = tok.apply_chat_template(
            [msgs[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=args.max_length)["input_ids"]
        plen = min(len(tok(prompt, add_special_tokens=False)["input_ids"]), len(ids))
        labels = [-100] * plen + ids[plen:]
        sup = tok.decode([t for t, l in zip(ids, labels) if l != -100])
        masked = tok.decode(ids[:plen])
        print(f"\n=== example {i} ===")
        print("SUPERVISED (loss on):", repr(sup))
        print("MASKED (no loss)    :", repr(masked[-160:]))


if __name__ == "__main__":
    main()
