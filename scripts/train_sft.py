"""SFT a Qwen3 model on FB15k-237 with QLoRA (4-bit base + LoRA adapters).

What this does, mapped to the Week 2 theory:
- Loads the base model in 4-bit (quantization, the "Q" in QLoRA) — frozen.
- Adds LoRA adapters (the only trainable params) on the attention projections.
- Builds both-direction chat examples (predict tail / predict head).
- Trains with trl's SFTTrainer, which applies Qwen3's chat template and (via
  `assistant_only_loss`) masks the loss to the assistant's answer tokens only.

Run on a GPU box / Kaggle. Smoke-test first to validate the pipeline cheaply:
    python scripts/train_sft.py --config configs/sft/qwen3_1.7b.yaml --max-triples 200
Then the full run:
    python scripts/train_sft.py --config configs/sft/qwen3_1.7b.yaml
"""

from __future__ import annotations

import os

# Use a SINGLE GPU. Kaggle "T4 x2" exposes two GPUs AND pre-sets
# CUDA_VISIBLE_DEVICES, so HF Trainer would wrap the model in DataParallel and clash
# with device_map="auto" -> "parameters on cuda:1". QLoRA of a 1.7B model fits on
# one T4, so we force the visible set down to its first entry (before importing
# torch). Must be a hard assignment — setdefault won't override Kaggle's value.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.llm.sft_data import make_sft_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--output-dir", default="artifacts/sft")
    ap.add_argument("--max-triples", type=int, default=None, help="override config (smoke runs)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    print(f"CUDA: {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    ds = load_fb15k237(args.data_dir)
    max_triples = args.max_triples if args.max_triples is not None else cfg.get("max_triples")
    train_ds = make_sft_dataset(ds, "train", max_triples=max_triples)
    print(f"SFT examples: {len(train_ds)} (from {max_triples or 'all'} triples, both directions)")

    # 4-bit quantization config (NF4 + double quant, fp16 compute — T4 has no bf16).
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        quantization_config=bnb,
        device_map="auto",
        # Force fp16. Qwen3 defaults to bf16, which leaves non-quantized params
        # (embeddings/norms/lm_head) in bf16 — and the fp16 grad scaler on a T4
        # can't unscale bf16 grads ("...not implemented for 'BFloat16'").
        dtype=torch.float16,
        attn_implementation="sdpa",  # PyTorch memory-efficient attention (faster than eager, T4-ok)
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.05),
        target_modules=cfg["lora"].get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA ourselves (instead of via SFTTrainer's peft_config) so we can force
    # the trainable adapter params to fp32. Qwen3's config dtype is bf16, which peft
    # otherwise propagates into the adapters — and the fp16 grad scaler on a T4 can't
    # unscale bf16 grads. Base stays 4-bit/frozen; only these fp32 adapters train.
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    model.print_trainable_parameters()

    out = Path(args.output_dir) / cfg["name"]
    sft_cfg = SFTConfig(
        output_dir=str(out),
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=float(cfg["train"]["lr"]),
        num_train_epochs=cfg["train"]["epochs"],
        max_length=cfg["train"].get("max_length", 256),
        packing=cfg["train"].get("packing", False),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        # No fp16 grad scaler. Qwen3's bf16 config leaks bf16 gradients, and the
        # fp16 scaler's CUDA unscale op isn't implemented for bf16 on a T4 (and the
        # T4 can't do bf16 training either). The 4-bit base already computes in fp16
        # via bnb and the adapters are fp32, so we skip autocast/scaling entirely —
        # stable for LoRA and it sidesteps the unsupported op.
        fp16=False,
        gradient_checkpointing=False,  # memory headroom (4-bit base, seq 128) -> ~30% faster
        assistant_only_loss=True,  # loss on the assistant answer tokens only
        report_to="wandb" if cfg.get("wandb") else "none",
        run_name=cfg["name"],
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,  # LoRA already applied above
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(str(out))  # saves the LoRA adapter
    print(f"\nSaved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
