# Kaggle runbook — KGE training (Phase 1, Day 3–5)

How to train the four KGE baselines on Kaggle's GPU and bring the results back.
Reusable for Phase 2 (Hetionet) — only the dataset and configs change.

Mental model: Kaggle gives you a Jupyter notebook running in a Linux container
with a GPU. You drive everything from notebook cells; shell commands run with a
leading `!`. Anything you write under `/kaggle/working/` and then *Save Version*
becomes downloadable output. `/kaggle/input/` is read-only attached data.

---

## Status — what to run now (as of Week 1, Day 6–7)

- **KGE baselines: DONE — skip §6–§10.** ComplEx, RotatE, TransE, QuatE are all
  trained and their bundles downloaded (filtered MRR: RotatE 0.324, QuatE 0.304,
  TransE 0.289, ComplEx 0.222). Don't re-run the KGE training/cross-check sections
  unless you're deliberately retuning a model — they're kept for reproducibility
  and Phase 2 (Hetionet).
- **New now: the zero-shot LLM baseline.** Reuse the same notebook setup (§1–§5),
  then jump to **"LLM zero-shot baseline"** at the bottom and run it for BOTH
  `Qwen/Qwen3-1.7B` and `Qwen/Qwen3.5-2B`.

---

## 0. One-time prerequisites

- **Phone-verify your Kaggle account** (Settings → Phone Verification). Required
  to enable both GPU and Internet in notebooks.
- Have your **W&B API key** ready (https://wandb.ai/authorize).

## 1. Put the data on Kaggle (once)

`data_cache/` is gitignored, so `git clone` will NOT bring the dataset. Upload it
as a Kaggle Dataset instead (reliable, fast, no internet needed at train time):

1. Kaggle → **Datasets → New Dataset**.
2. Upload the five files from your local `data_cache/fb15k237/`:
   `train.tsv`, `valid.tsv`, `test.tsv`, `entity2text.txt`, `relation2text.txt`.
3. Name it `fb15k237`. After creation the files live at
   `/kaggle/input/fb15k237/`.

(Alternative: enable Internet and run `python scripts/download_fb15k237.py` in the
notebook. The uploaded-dataset route is more reproducible — pin it.)

## 2. Create the notebook and set resources

1. Kaggle → **Create → Notebook**.
2. Right-hand **Settings** panel:
   - **Accelerator:** `GPU T4 x2` (we only use one) or `P100`.
   - **Internet:** **On** (needed for `git clone`, `pip`, and W&B sync).
   - **Environment:** latest / pin to "Always use latest".
3. **+ Add Input** → search and attach your `fb15k237` dataset.
4. **Add-ons → Secrets** → add `WANDB_API_KEY` = your key, and toggle it on for
   this notebook.

## 3. Setup cell (clone + install)

```python
!git clone <YOUR_REPO_URL> kgllm
%cd kgllm
# pykeen is not preinstalled; torch IS (with CUDA) — do NOT reinstall torch.
!pip install -q pykeen pyyaml wandb
# register our package WITHOUT touching Kaggle's CUDA torch:
!pip install -q -e . --no-deps
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

**Verify `cuda available: True` before training.** If it prints False, restart the
session (the install shouldn't change torch because of `--no-deps`, but confirm).

## 4. W&B auth cell

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

## 5. Sanity-check data + loader

```python
!ls -la /kaggle/input/fb15k237
!python -c "from kg_llm.data.fb15k237 import load_fb15k237; \
ds=load_fb15k237('/kaggle/input/fb15k237'); print(ds.num_entities, ds.num_relations)"
# expect: 14541 237
```

## 6. Train — ComplEx first (validate the full loop before batching)

```python
!python scripts/train_kge.py --config configs/kge/complex.yaml \
    --data-dir /kaggle/input/fb15k237 \
    --output-dir /kaggle/working/artifacts
```

Watch the progress bar say **`Training epochs on cuda`** with **seconds**, not
minutes, per epoch. Note the wall-clock time — that tells you whether all four fit
in one ~9h session.

## 7. Verify in the cloud (cross-check our harness)

```python
!python scripts/crosscheck_kge_eval.py \
    --model-dir /kaggle/working/artifacts/complex_fb15k237 \
    --data-dir /kaggle/input/fb15k237
```

Expect **"OK — harnesses agree within tolerance."** and an MRR in the
**~0.30–0.32** ballpark (a little under the published 0.32 is fine — we used
embedding_dim 256, not the paper's larger dims). If MRR is wildly off or the
assert fails, stop and paste the output before training the rest.

## 8. Package outputs for download

```python
!cd /kaggle/working && tar czf complex_fb15k237.tgz -C artifacts complex_fb15k237
!ls -lah /kaggle/working/*.tgz
```

Then **Save Version** (top-right) → *Save & Run All (Commit)* or *Quick Save*.
When it finishes, open the notebook's **Output** tab → `/kaggle/working` → download
the `.tgz`.

## 9. Train the other three

Repeat steps 6–8 with `configs/kge/transe.yaml`, `rotate.yaml`, `quate.yaml`
(output dirs `transe_fb15k237/`, `rotate_fb15k237/`, `quate_fb15k237/`). If timing
from step 6 shows it fits, run all three in the same session sequentially. Anchor
to clear: RotatE should be the strongest (~0.33–0.34).

## 10. What to bring back local (per model)

From each `<name>.tgz`, the files that matter:

| File | Keep? | Why |
|------|-------|-----|
| `embeddings.pt` | **Yes — required** | Raw vectors (on our IDs) for Week 3 hard-negative mining. |
| `results.json` | **Yes** | The filtered metrics → goes into the results table. |
| `trained_model.pkl` | Optional | Only if you want to re-evaluate locally later; large, and we already cross-checked in cloud. |
| `training_triples/`, W&B files | No | Reproducible / already in the cloud. |

Locally, drop them under `artifacts/kge/<name>/` (gitignored). Then **record the
numbers** into the tracked files: the README results table and
`reports/writeup_notes.md`. Those commits are the portfolio-visible artifact; the
big binaries stay out of git.

## LLM zero-shot baseline (Week 1, Day 6–7) — the new run

This is the only new Kaggle work. It reuses the same notebook setup as the KGE runs
(§2–§5: GPU on, internet on, `fb15k237` dataset attached, repo cloned and installed
with `pip install -e . --no-deps`). Two additions:

```python
# Qwen3.5 is very new — make sure transformers is recent enough to load it.
!pip install -q -U transformers accelerate
```

The model weights download from Hugging Face (~3–4 GB each), so internet must be on.

First, validate the scoring math locally (CPU, no GPU needed) — ideally on your own
machine before even opening Kaggle:

```
pytest tests/test_llm_scorer.py -q
```

Then in the notebook:

```python
# 1) tiny dry run per model — confirms loading + scoring, and times one model
!python scripts/eval_llm_zeroshot.py --model Qwen/Qwen3-1.7B --data-dir /kaggle/input/fb15k237 --num-test 5
!python scripts/eval_llm_zeroshot.py --model Qwen/Qwen3.5-2B --data-dir /kaggle/input/fb15k237 --num-test 5

# 2) the real baselines (use the dry-run timing to confirm 1000 fits the ~9h
#    session; lower --num-test if not). Same --seed => same test subset for both.
!python scripts/eval_llm_zeroshot.py --model Qwen/Qwen3-1.7B --data-dir /kaggle/input/fb15k237 --num-test 1000
!python scripts/eval_llm_zeroshot.py --model Qwen/Qwen3.5-2B --data-dir /kaggle/input/fb15k237 --num-test 1000
```

- **Sampled ranking.** Scoring all 14,541 entities per query is infeasible on a
  T4, so the gold is ranked against `--num-candidates` candidates (default 256 =
  gold + 255 sampled negatives). This makes a 1000-triple run take minutes. The
  metric is over 256 candidates, so it is NOT directly comparable to KGE's
  full-14,541 MRR — fine for the zero-shot floor; for a head-to-head later we run
  KGE under the same sampling. Raise `--num-candidates` (e.g. 1000) for fidelity at
  more cost; lower `--cand-batch-size` if you hit OOM.
- **Check the device print.** The script prints `CUDA available: ...` and
  `model device: ...` — if it says CPU, your notebook's GPU accelerator is off.
- **Output:** the script only prints a metrics line — no big files to download.
  Copy the `MRR=… H@1=… H@3=… H@10=…` line for each model and paste it back; both
  go into the README table. Expect both **below the 0.23 frequency floor** — that's
  the zero-shot floor SFT/DPO will climb from, not a bug.

### If Qwen3.5-2B crashes (`CUDA error: unspecified launch failure`)

Qwen3.5 is a linear-attention model; without its kernels it runs an unstable
pure-torch fallback (you'll see "The fast path is not available ... Falling back to
torch implementation"). Try, in order:

```python
# 1) install the linear-attention kernels so it uses the stable fast path
!pip install -q flash-linear-attention causal-conv1d

# 2) if it still crashes, get the REAL error location (turns async into sync)
import os; os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# ...then re-run the Qwen3.5 command and paste the new traceback

# 3) or just shrink the batch — sometimes avoids the fallback's bad path
!python scripts/eval_llm_zeroshot.py --model Qwen/Qwen3.5-2B --data-dir /kaggle/input/fb15k237 --num-test 1000 --cand-batch-size 64
```

Qwen3-1.7B (standard attention) is the stable anchor; if Qwen3.5 stays flaky,
report Qwen3-1.7B and note the instability.

## Quotas & gotchas

- GPU session ≈ 9h; weekly GPU quota ≈ 30h (your AIMO boost helps). Doing ComplEx
  alone first lets you time the rest.
- Internet **off** → `git clone` / `pip` / W&B all fail. Turn it on.
- Never `pip install torch` on Kaggle — it ships a CUDA build; `--no-deps` keeps it.
- Output persists only if it's under `/kaggle/working/` **and** you Save Version.
