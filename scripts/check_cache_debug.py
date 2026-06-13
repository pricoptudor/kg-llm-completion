"""Localize the prompt-KV-cache bug: compare per-token candidate log-probs from the
uncached reference vs several cached variants (different position handling), for one
query and one candidate. The variant whose per-token log-probs match 'nocache' is the
correct API to use.

    python scripts/check_cache_debug.py --model <path> --data-dir <fb15k237>
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kg_llm.data.fb15k237 import load_fb15k237


def nocache(model, prefix_ids, cand_ids, dev):
    ids = torch.tensor([prefix_ids + cand_ids], device=dev)
    logits = model(input_ids=ids).logits.float()[0]
    plen, a = len(prefix_ids), len(cand_ids)
    pred = logits[plen - 1 : plen - 1 + a]
    lp = torch.log_softmax(pred, -1)
    return lp[torch.arange(a), torch.tensor(cand_ids, device=dev)]


def cached(model, prefix_ids, cand_ids, dev, mode):
    plen, a = len(prefix_ids), len(cand_ids)
    pout = model(input_ids=torch.tensor([prefix_ids], device=dev), use_cache=True)
    cache, last = pout.past_key_values, pout.logits.float()[:, -1, :]
    cand_t = torch.tensor([cand_ids], device=dev)
    attn = torch.ones(1, plen + a, device=dev).long()
    kw = dict(past_key_values=cache, use_cache=True)
    if mode == "cache_position":
        kw["attention_mask"] = attn
        kw["cache_position"] = torch.arange(plen, plen + a, device=dev)
    elif mode == "position_ids":
        kw["attention_mask"] = attn
        kw["position_ids"] = torch.arange(plen, plen + a, device=dev).unsqueeze(0)
    elif mode == "both":
        kw["attention_mask"] = attn
        kw["cache_position"] = torch.arange(plen, plen + a, device=dev)
        kw["position_ids"] = torch.arange(plen, plen + a, device=dev).unsqueeze(0)
    elif mode == "attn_only":
        kw["attention_mask"] = attn
    elif mode == "none":
        pass
    logits_cand = model(input_ids=cand_t, **kw).logits.float()[0]
    pred = torch.cat([last, logits_cand[: a - 1]], dim=0)
    lp = torch.log_softmax(pred, -1)
    return lp[torch.arange(a), torch.tensor(cand_ids, device=dev)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    args = ap.parse_args()

    ds = load_fb15k237(args.data_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto").eval()

    h, r, t = ds.test_triples[0].tolist()
    prefix = f"Head: {ds.entity_name(h)}. Relation: {ds.relation_name(r)}. Tail:"
    prefix_ids = tok(prefix, add_special_tokens=True).input_ids
    cand_ids = tok(" " + ds.entity_name(t), add_special_tokens=False).input_ids
    print(f"prefix={prefix!r}\ncand={ds.entity_name(t)!r} -> {cand_ids}  (plen={len(prefix_ids)}, a={len(cand_ids)})\n")

    with torch.no_grad():
        ref = nocache(model, prefix_ids, cand_ids, dev)
        print("nocache  (reference):", [f"{x:.4f}" for x in ref.tolist()])
        for mode in ("cache_position", "position_ids", "both", "attn_only", "none"):
            try:
                got = cached(model, prefix_ids, cand_ids, dev, mode)
                d = (got - ref).abs().max().item()
                tag = "  <== MATCH" if d < 1e-2 else ""
                print(f"cached[{mode:14s}]:", [f"{x:.4f}" for x in got.tolist()], f" max|Δ|={d:.2e}{tag}")
            except Exception as e:  # noqa: BLE001
                print(f"cached[{mode:14s}]: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
