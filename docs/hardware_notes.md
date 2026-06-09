# Hardware notes — two stacks (T4 vs capable GPU), and why

We have two working stacks because the GPU changed mid-project. Keep BOTH: the
capable-GPU stack is the default now; the T4 stack is the fallback if we're ever
back on a weak/old GPU.

## Capable GPU (compute >= 8.0, bf16) — e.g. Kaggle RTX Pro 6000 — CURRENT DEFAULT

- **Train:** `scripts/train_sft.py` — plain transformers + peft + bitsandbytes +
  trl SFTTrainer, **bf16**, QLoRA. No Unsloth. Clean: bf16 means no grad scaler, so
  none of the fp16/bf16 hacks are needed. `--model <local path>` for offline.
- **Eval:** `scripts/eval_llm_zeroshot.py` (HF stack, `--full` for full-candidate)
  or `scripts/eval_llm_vllm.py` (vLLM, faster, but needs wheels staged offline).
- **Why it's simpler:** the strong GPU removes every workaround below.

## T4 (compute 7.5, NO bf16) — FALLBACK — `scripts/train_sft_unsloth_t4.py`

Hard-won lessons from training/evaluating Qwen3 QLoRA on a free Kaggle T4. If we
ever go back to a T4 (or similar Turing/Pascal card), these are required:

1. **Unsloth, not plain trl.** Plain trl + bitsandbytes on a T4 hit a wall: Qwen3's
   bf16 config crashes the fp16 grad scaler (`_amp_foreach...not implemented for
   BFloat16`), and 4-bit dequant without fp16 Tensor Cores is slow. Unsloth fixes
   both (correct T4 fp16 + ~2x kernels). Its SFTTrainer needs the dataset rendered
   to a `text` field (not raw `messages`) and masking via `train_on_responses_only`.
2. **fp16, never bf16** on Turing (T4 has no bf16). If forced to plain trl, you must
   load the model fp16, cast LoRA adapters to fp32, and even then often `fp16=False`
   to dodge the scaler entirely.
3. **No `packing`** without flash-attention (Turing lacks FA2): packing flattens
   samples and, without FA2, cross-contaminates them (token-acc 0.58 vs 0.66) AND
   runs slower. Confirmed empirically.
4. **No vLLM on T4.** FA2 needs compute >= 8.0, so vLLM falls back to FlashInfer,
   whose JIT kernel fails to link (`cannot find -lcuda`), and the version's
   `VLLM_ATTENTION_BACKEND` knob was removed. Not worth fighting — use the HF eval.
5. **Single GPU pin** (`CUDA_VISIBLE_DEVICES=0`): Kaggle "T4 x2" + HF Trainer's
   DataParallel clashes with `device_map="auto"`.
6. **Speed levers that helped:** disable gradient checkpointing (memory headroom),
   `attn_implementation="sdpa"`, bigger per-device batch. `group_by_length` is not
   in this trl's SFTConfig. Free VRAM != free speed (we were compute-bound at b32).
7. **Eval shortcut:** 256-way *sampled* ranking (gold + 255 negatives) instead of
   full 14,541 — a dev metric only, NOT comparable to KGE/SOTA. The capable GPU
   lets us do full-candidate (`--full`), which IS comparable.

## Offline Kaggle (GPU session has no internet) — applies to BOTH stacks

Everything network-bound must be pre-staged; the GPU session runs fully offline.
- Models: attach the Qwen-3 **Kaggle Model** (offline), pass its local path to
  `--model`. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- Packages: prefer preinstalled (transformers/peft); stage wheels via `pip download`
  in a CPU+internet session -> upload as a Dataset -> `pip install --no-index
  --find-links`. (This is why we favor the HF stack over vLLM/Unsloth offline.)
- Code: upload the repo as a Dataset (no `git clone` offline), `pip install -e .
  --no-deps`.
- Data / adapters: Kaggle Datasets (offline-ready).
- W&B: `WANDB_MODE=offline` (sync later) or `report_to="none"`.
