"""SFT a Qwen3 model on FB15k-237 with QLoRA via Unsloth (fast 4-bit + LoRA).

Unsloth replaces the plain transformers/bitsandbytes model loading with custom
Triton kernels: ~2x faster training, much lower memory, and — crucially for the
T4 — it manages fp16 precision correctly, so we can re-enable fp16 (which the plain
path couldn't, due to Qwen3's bf16 config crashing the grad scaler). Everything
else is unchanged: both-direction chat examples, answer-only loss masking, our YAML
config. A GPUMonitor callback prints VRAM + utilization so we can see headroom.

Run on a GPU box / Kaggle (smoke first):
    python scripts/train_sft.py --config configs/sft/qwen3_1.7b.yaml --max-triples 200
    python scripts/train_sft.py --config configs/sft/qwen3_1.7b.yaml
"""

from __future__ import annotations

import os

# Use ONE GPU (Kaggle "T4 x2" pre-sets CUDA_VISIBLE_DEVICES; reduce to its first).
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"

from unsloth import FastLanguageModel  # noqa: E402  (must import before transformers/trl)

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import TrainerCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from kg_llm.data.fb15k237 import load_fb15k237  # noqa: E402
from kg_llm.llm.sft_data import make_sft_dataset  # noqa: E402


class GPUMonitor(TrainerCallback):
    """Print GPU memory (and utilization, if pynvml is available) at each log."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        util = ""
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = f"  util={pynvml.nvmlDeviceGetUtilizationRates(h).gpu}%"
            pynvml.nvmlShutdown()
        except Exception:
            pass
        print(f"[GPU] VRAM reserved={reserved:.1f}/{total:.0f}GB  peak={peak:.1f}GB{util}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--output-dir", default="artifacts/sft")
    ap.add_argument("--max-triples", type=int, default=None, help="override config (smoke runs)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    ds = load_fb15k237(args.data_dir)
    max_triples = args.max_triples if args.max_triples is not None else cfg.get("max_triples")
    train_ds = make_sft_dataset(ds, "train", max_triples=max_triples)
    print(f"SFT examples: {len(train_ds)} (from {max_triples or 'all'} triples, both directions)")

    max_len = cfg["train"].get("max_length", 128)
    # Unsloth: 4-bit load + auto dtype (fp16 on a T4) + its own fast kernels.
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["model"],
        max_seq_length=max_len,
        dtype=None,  # auto-detect (fp16 on T4)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        target_modules=cfg["lora"].get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.0),
        bias="none",
        # Unsloth's gradient checkpointing is memory-efficient WITHOUT the usual
        # speed penalty — frees VRAM for a bigger batch.
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    out = Path(args.output_dir) / cfg["name"]
    sft_cfg = SFTConfig(
        output_dir=str(out),
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=float(cfg["train"]["lr"]),
        num_train_epochs=cfg["train"]["epochs"],
        max_length=max_len,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        fp16=True,  # Unsloth handles T4 fp16 correctly (no bf16 grad-scaler crash)
        assistant_only_loss=True,  # loss on the assistant answer tokens only
        report_to="wandb" if cfg.get("wandb") else "none",
        run_name=cfg["name"],
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        processing_class=tok,
        callbacks=[GPUMonitor()],
    )
    trainer.train()
    trainer.save_model(str(out))
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Peak VRAM: {torch.cuda.max_memory_reserved()/1e9:.1f} GB of {total:.0f} GB "
              f"(headroom => room for a bigger batch)")
    print(f"\nSaved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
