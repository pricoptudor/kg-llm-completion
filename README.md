# KG-LLM Completion

**When do LLMs beat classical KGE for knowledge graph completion?**
An empirical study with post-training (SFT + DPO) and KGE-mined hard negatives.

> Work in progress. Phase 1 ships at Week 4 (FB15k-237); Phase 2 at Week 8 (Hetionet).

## Why this project

Knowledge-graph completion (link prediction) has historically been dominated by embedding methods — ComplEx, RotatE, QuatE — that score `(head, relation, tail)` triples in geometric space. Modern LLMs offer a different lever: they bring world knowledge from pretraining. This project measures, on standard benchmarks, where each paradigm wins, and proposes a recipe — *KGE-mined hard negatives for DPO* — that combines the strengths of both.

## Results

Filtered MRR / Hits@k on FB15k-237 test split (head+tail pooled). KGE metrics are
from PyKEEN's evaluator; our independent harness is cross-checked against it and
agrees exactly (validated on the non-inverse models). These are **dim-256**
baselines — competitive with published numbers despite the small dimension, except
ComplEx, which needs a much larger dim to reach its ~0.32 anchor.

| Method                          | MRR   | H@1   | H@3   | H@10  |
| ------------------------------- | ----- | ----- | ----- | ----- |
| Frequency baseline              | 0.233 | 0.170 | 0.250 | 0.354 |
| TransE                          | 0.289 | 0.195 | 0.324 | 0.476 |
| ComplEx                         | 0.222 | 0.154 | 0.242 | 0.358 |
| RotatE                          | **0.324** | **0.229** | **0.361** | **0.518** |
| QuatE                           | 0.304 | 0.217 | 0.335 | 0.483 |
| Qwen3-1.7B (zero-shot)          | —     | —     | —     | —     |
| Qwen3-1.7B SFT                  | —     | —     | —     | —     |
| Qwen3-1.7B SFT + DPO (random)   | —     | —     | —     | —     |
| Qwen3-1.7B SFT + DPO (KGE)      | —     | —     | —     | —     |

RotatE is the strongest KGE baseline; every method clears the frequency floor by a
clear margin. The LLM side spans a **model-scaling axis** (Qwen3 0.6B / 1.7B / 4B)
across zero-shot → SFT → DPO; final LLM numbers use full-candidate filtered ranking
(via vLLM) to stay comparable to the KGE table. Phase 2 (Hetionet) forthcoming.

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
