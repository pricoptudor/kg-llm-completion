# KG-LLM Completion

**When do LLMs beat classical KGE for knowledge graph completion?**
An empirical study with post-training (SFT + DPO) and KGE-mined hard negatives.

> Work in progress. Phase 1 ships at Week 4 (FB15k-237); Phase 2 at Week 8 (Hetionet).

## Why this project

Knowledge-graph completion (link prediction) has historically been dominated by embedding methods — ComplEx, RotatE, QuatE — that score `(head, relation, tail)` triples in geometric space. Modern LLMs offer a different lever: they bring world knowledge from pretraining. This project measures, on standard benchmarks, where each paradigm wins, and proposes a recipe — *KGE-mined hard negatives for DPO* — that combines the strengths of both.

## Results

Filtered MRR / Hits@k on FB15k-237 test split (head+tail pooled). KGE metrics are
from PyKEEN's evaluator; our independent harness is cross-checked against it and
agrees exactly on the non-inverse models. These are dim-256 starting baselines —
a bit under published numbers (which use much larger dims) and still being tuned.

| Method                          | MRR   | H@1   | H@3   | H@10  |
| ------------------------------- | ----- | ----- | ----- | ----- |
| Frequency baseline              | 0.233 | 0.170 | 0.250 | 0.354 |
| TransE                          | 0.289 | 0.195 | 0.324 | 0.476 |
| ComplEx                         | _re-run pending_ | | | |
| RotatE                          | _retuning_ | | | |
| QuatE                           | _pending_ | | | |
| Qwen2.5-1.5B (zero-shot)        | —     | —     | —     | —     |
| Qwen2.5-1.5B SFT                | —     | —     | —     | —     |
| Qwen2.5-1.5B SFT + DPO (random) | —     | —     | —     | —     |
| Qwen2.5-1.5B SFT + DPO (KGE)    | —     | —     | —     | —     |

Notes: ComplEx must be re-run with the current LCWA+inverse config (the earlier
artifact predated that fix; prelim MRR ≈ 0.22). RotatE was undertrained at dim
256 / lr 1e-4 and is retuning (lr 5e-4, 128 negs, 150 epochs). Phase 2 (Hetionet)
table forthcoming.

## Reproduction

```bash
git clone https://github.com/<user>/kg-llm-completion
cd kg-llm-completion
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                              # add ",train" on Kaggle/Colab

python scripts/download_fb15k237.py                  # ~10 MB, idempotent
python scripts/train_kge.py --config configs/kge/complex.yaml
# ... full reproduction instructions filled in as scripts land
```

## Layout

```
src/kg_llm/    Installable package: data loaders, KGE wrappers, LLM trainers, eval
scripts/       CLI entry points
configs/       YAML configs (Hydra-composable)
notebooks/     EDA and analysis
tests/         Unit tests for the bits that have to be right (filtered eval indexer, etc.)
reports/       Writeups, figures
```

## Method (brief)

1. Train classical KGE baselines (ComplEx, RotatE, TransE, QuatE) with PyKEEN.
2. SFT Qwen2.5-1.5B-Instruct on FB15k-237 triples reformatted as natural-language Q→A pairs.
3. Mine hard negatives using KGE embeddings: for each true `(h, r, t)`, find an entity `t'` whose embedding is close to `t` but which does not actually form a triple with `(h, r)`.
4. Run DPO with `(prompt, t, t')` preference pairs.
5. Evaluate filtered Hits@k / MRR; ablate negative-mining strategies; analyse per-relation winners.

## Citation

```
Pricop, T. (2026). When do LLMs beat classical KGE for knowledge graph completion?
Work in progress.
```

## License

MIT. See `LICENSE` (to add).
