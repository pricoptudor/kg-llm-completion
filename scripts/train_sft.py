"""SFT a Qwen3 model on FB15k-237 with QLoRA (4-bit base + LoRA), bf16.

Target: a capable GPU (compute >= 8.0, bf16) such as Kaggle's RTX Pro 6000, run
OFFLINE. Plain transformers + peft + bitsandbytes (NO trl, NO Unsloth). Answer-only
masking is done manually (tokenize the chat, set prompt-token labels to -100), so it
doesn't depend on any trl masking feature or `{% generation %}` template tags. The
Unsloth/T4 variant is preserved at scripts/train_sft_unsloth_t4.py
(rationale in docs/hardware_notes.md).

Offline: set HF_HUB_OFFLINE=1 and pass --model <local path>. bf16 auto-detected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from kg_llm.data.fb15k237 import load_fb15k237
from kg_llm.llm.sft_data import make_sft_dataset


class GPUMonitor(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not torch.cuda.is_available():
            return
        res = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[GPU] VRAM reserved={res:.1f}/{total:.0f}GB  peak={peak:.1f}GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data_cache/fb15k237")
    ap.add_argument("--output-dir", default="artifacts/sft")
    ap.add_argument("--model", default=None, help="override base model path (local dir for offline)")
    ap.add_argument("--max-triples", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    model_name = args.model or cfg["model"]
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"CUDA={torch.cuda.is_available()}  bf16={bf16}  base={model_name}")

    ds = load_fb15k237(args.data_dir)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map="auto", dtype=compute_dtype
    )
    gc = bool(cfg["train"].get("gradient_checkpointing", False))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gc)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"].get("dropout", 0.05),
            target_modules=cfg["lora"].get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    # Build both-direction chat examples, then tokenize with answer-only masking:
    # labels = -100 for the prompt tokens (everything up to "...assistant\n"),
    # real ids for the answer tokens.
    max_triples = args.max_triples if args.max_triples is not None else cfg.get("max_triples")
    raw = make_sft_dataset(ds, "train", max_triples=max_triples)
    maxlen = cfg["train"].get("max_length", 128)

    def tok_mask(ex):
        msgs = ex["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        prompt = tok.apply_chat_template(
            [msgs[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(full, add_special_tokens=False, truncation=True, max_length=maxlen)["input_ids"]
        plen = min(len(tok(prompt, add_special_tokens=False)["input_ids"]), len(ids))
        labels = [-100] * plen + ids[plen:]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    train_ds = raw.map(tok_mask, remove_columns=raw.column_names)
    print(f"SFT examples: {len(train_ds)} (from {max_triples or 'all'} triples, both directions)")

    collator = DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100)

    out = Path(args.output_dir) / cfg["name"]
    targs = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=float(cfg["train"]["lr"]),
        num_train_epochs=cfg["train"]["epochs"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        save_strategy="epoch",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=gc,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if cfg.get("wandb") else "none",
        run_name=cfg["name"],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tok,
        callbacks=[GPUMonitor()],
    )
    trainer.train()
    trainer.save_model(str(out))
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Peak VRAM: {torch.cuda.max_memory_reserved()/1e9:.1f} GB of {total:.0f} GB")
    print(f"\nSaved LoRA adapter to {out}")


if __name__ == "__main__":
    main()
