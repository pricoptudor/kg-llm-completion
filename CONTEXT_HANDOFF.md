# Conversation handoff — KG_LLM_Completion

This document is a checkpoint for any future Claude conversation picking up
work on this project. The authoritative plan is `kg_llm_project_plan.md`;
this file captures **where we are inside that plan** as of the most recent
session, plus the working agreements that aren't visible from code.

> Last updated: 2026-05-18 (session immediately after the FB15k-237 download
> script's smoke check passed on Tudor's Windows machine).

---

## Working agreement (do this every session)

- **Pair-programming mode.** Claude writes code directly into the workspace
  folder with step-by-step explanations. Tudor reviews and makes architectural
  changes / pushes back when warranted.
- **Theory depth: intuition + key equations.** Not full derivations from first
  principles, but no hand-waving either. Reference the relevant paper.
- **Honest pushback.** If Tudor is about to do something wrong or suboptimal,
  say so directly. Concrete over abstract advice.
- **Git lives on Tudor's machine.** The sandbox cannot reach GitHub (proxy
  blocks `raw.githubusercontent.com`, `huggingface.co`, etc.) and cannot
  manage Windows-mounted `.git/` files reliably. All `git`, `gh`, and `pip`
  commands run from Tudor's Windows PowerShell.
- **Tudor's compute:** currently on a no-GPU laptop, will move to GTX 1650
  (4GB, dev only) and Kaggle T4 (16GB, AIMO-boosted quota) for actual training.
- **Use TaskCreate/TaskUpdate** to track progress. See "Open tasks" below.

---

## Where we are in the plan

We are in **Phase 1, Week 1, Day 1–2** of the 8-week plan.

### Done

- [x] Conceptual foundations covered: what a KG is, link prediction, closed-world
      assumption, FB15k vs FB15k-237 (and why removing inverse-relation leakage
      makes the benchmark harder), the four KGE baselines (TransE, ComplEx,
      RotatE, QuatE) with scoring functions, ComplEx symmetric/antisymmetric
      analysis via the imaginary part of the relation embedding, why filtered
      evaluation is necessary (1-to-N relations punish models for being right).
- [x] Repo scaffolding written: `pyproject.toml` (installable package, optional
      `[dev]`/`[train]`/`[eval]` extras), `.gitignore`, `README.md` (with
      placeholder results table), `.env.example`, empty `src/kg_llm/<subpkg>/__init__.py`
      stubs for `data`, `kge`, `llm`, `eval`, `utils`.
- [x] `git init -b main` on Windows, first commit pushed to GitHub via
      `gh repo create kg-llm-completion --public --source=. --push`. Repo
      is public from day one (intentional — portfolio signal).
- [x] `pip install -e ".[dev]"` succeeded inside `.venv`.
- [x] `scripts/download_fb15k237.py` written and **smoke-check passing**
      after one fix-up round (see "Lessons" below). Files live in
      `data_cache/fb15k237/`.

### Open tasks (current session was paused here)

The TaskCreate list at the time of the handoff:

| # | Status        | Title                                                | Notes |
|---|---------------|------------------------------------------------------|-------|
| 1 | completed     | Conceptual foundations: KGs, link prediction, FB15k-237 | |
| 2 | completed     | Write repo scaffolding + git init                    | |
| 3 | in_progress   | Download FB15k-237 and inspect                       | Smoke check passing; waiting on Tudor's explicit "signal" before marking complete and moving on. |
| 4 | pending       | Build data loading module                            | `src/kg_llm/data/fb15k237.py`. Needs `(h, r, t)` int tensors + id↔name maps + filtered-eval index. |
| 5 | pending       | Concept + implementation: filtered evaluation        | Cover the math properly, then implement against a brute-force scorer on a toy. |
| 6 | pending       | Verify Day 1–2 setup end-to-end                      | Smoke test the harness with a trivial baseline (e.g., frequency of tail given relation). |

---

## Architectural decisions made (and why)

These were chosen deliberately; if they need to change, change them on
purpose — don't drift into something else by accident.

1. **`src/kg_llm/` package layout** (not top-level modules). `pip install -e .`
   makes imports clean across scripts and notebooks. No `sys.path` hacks.
2. **`pyproject.toml`** over bare `requirements.txt`. Modern packaging; lets
   us split deps via optional extras (`[dev]`, `[train]`, `[eval]`).
   Versions are floors, not pins — we lock with `pip freeze > requirements.lock`
   once Week 1 is green.
3. **Hydra for configs** (declared in deps; not used yet). Justified because
   we'll hit 30+ experiments by Week 4 (4 KGE methods, SFT, 3 negative-mining
   strategies for DPO, β-ablations, ...).
4. **W&B project naming**: single project `kg-llm-completion`, tagged by
   `{phase, method, dataset, neg_strategy}`. One project so Phase 1 and
   Phase 2 curves can overlay in the writeup.
5. **FB15k-237 source: KG-BERT repo (`yao8839836/kg-bert`)** for both triples
   and text labels. Originally tried `TimDettmers/ConvE` — that repo no
   longer hosts loose `train.txt`/`valid.txt`/`test.txt` files (now ships
   a tarball), so the URL 404s. KG-BERT republishes the canonical splits
   and ships the text labels we need. One source, five files.
6. **`/m/...` Freebase MID prefix kept verbatim** as the entity ID; we don't
   strip it. Matters because some downstream tools strip and some don't —
   if we ever ingest mappings from multiple sources we'll need a normalizer,
   but for now stick with verbatim.
7. **Public GitHub repo from day one** (`gh repo create --public`). Portfolio
   signal: hiring managers see incremental honest commits, not a long
   polishing window followed by one mega-commit.

---

## Lessons from this week (write into the writeup later)

- **FB15k vs FB15k-237 entity-count gotcha.** KG-BERT's `entity2text.txt`
  inside their FB15k-237 folder has **14,951** entries — the original FB15k
  count — not 14,541. The extra 410 entries are orphans (entities that only
  appeared in relations FB15k-237 dropped). The file is therefore a *superset*,
  not a corruption. Our smoke check originally did `count == 14_541` and
  flagged this as a failure; we replaced strict equality with a **coverage
  check**: every entity in `train+valid+test` must have a label, but the
  label file may have extras. The triple-side counts (272,115 / 17,535 /
  20,466 / 14,541 entities / 237 relations) are still checked strictly.

- **Sandbox proxy is locked down.** The Linux sandbox can't reach
  `raw.githubusercontent.com` or `huggingface.co`. Anything network-bound
  has to run on Tudor's Windows side. Don't try to download from the sandbox.

---

## Conceptual context already covered

(For the new session to know what NOT to re-explain unless asked.)

- Definition of a KG as $\mathcal{G} \subset \mathcal{E} \times \mathcal{R} \times \mathcal{E}$,
  directed and multi-relational.
- Closed-world assumption and why we live with it.
- Link prediction as ranking candidates for $(h, r, ?)$ or $(?, r, t)$ queries.
- FB15k → FB15k-237 history (Toutanova & Chen 2015 removed inverse-relation
  leakage); why the new benchmark is harder.
- TransE: $f = -\|h + r - t\|$; can't model symmetric/1-to-N.
- ComplEx: $f = \mathrm{Re}(\sum_i h_i r_i \bar{t_i})$; complex conjugate breaks
  symmetry; relation's imaginary part dials between symmetric and antisymmetric.
- RotatE: $f = -\|h \odot r - t\|$ with $|r_i| = 1$; rotations in $\mathbb{C}^d$.
- QuatE: quaternion generalization; connects to Tudor's PhD Clifford-algebra
  direction.
- Why filtered evaluation: 1-to-N relations (e.g., Einstein's many awards)
  shouldn't penalize a correct model — filter out alternative-correct
  $(h, r, t')$ from the candidate set, leaving only the test target competing
  against wrong answers.
- LLM scoring formulation: $\log p_\theta(t \mid \text{prompt})$ over candidates,
  not generation. Apples-to-apples with KGE ranking.

## Conceptual context still to cover (queued)

- The *math* of filtered evaluation: MRR formula, Hits@k formula, how to
  average over head and tail prediction. (Task 5, coming next.)
- KGE training loss (negative sampling, NSSA — Sun et al. 2019); margin loss
  vs cross-entropy with negatives.
- SFT loss masking (only on assistant turn).
- DPO loss derivation from RLHF: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}[\log \sigma(\beta (\log \pi_\theta(y_w|x)/\pi_\text{ref}(y_w|x) - \log \pi_\theta(y_l|x)/\pi_\text{ref}(y_l|x)))]$.
- Hard negative mining for DPO — the "clever bit" of the project.
- Catastrophic forgetting / alignment tax measurement.

---

## Files of interest

| Path                                  | Purpose                                            |
|---------------------------------------|----------------------------------------------------|
| `kg_llm_project_plan.md`              | Authoritative 8-week plan. Always re-read first.   |
| `pyproject.toml`                      | Deps + ruff config.                                |
| `scripts/download_fb15k237.py`        | Idempotent dataset downloader with smoke check.    |
| `data_cache/fb15k237/`                | train.tsv / valid.tsv / test.tsv / entity2text.txt / relation2text.txt. Gitignored. |
| `src/kg_llm/data/__init__.py`         | Empty stub; data loader lands here next (Task 4).  |
| `CONTEXT_HANDOFF.md`                  | This file.                                         |

---

## Next concrete action when picking up

1. Re-read this file and `kg_llm_project_plan.md` Week 1 Day 6–7 onward.
2. Confirm with Tudor that he wants to proceed (he asked for an explicit
   signal before Task 4 last session).
3. Move to **Task 4 — Build data loading module** in `src/kg_llm/data/fb15k237.py`.
   It should expose:
   - `load_fb15k237(data_dir) -> FB15k237` returning a dataclass with
     `train_triples`, `valid_triples`, `test_triples` as `LongTensor`s of
     shape `(N, 3)`, plus `entity_to_id`, `id_to_entity`, `id_to_name`,
     `relation_to_id`, `id_to_relation`, `id_to_relation_name`.
   - A `FilteredIndex` helper for filtered evaluation (Task 5).
4. Then **Task 5 — filtered evaluation theory + implementation** in
   `src/kg_llm/eval/ranking.py`, verified against a brute-force scorer on a
   toy KG inside `tests/`.
5. **Task 6 — end-to-end smoke**: a trivial baseline that ranks tail
   candidates by relation-conditional frequency, just to prove the harness
   produces sensible numbers.

Once Tasks 4–6 land, Week 1 Day 1–2 of the plan is genuinely done and we
move into Day 3–5 (PyKEEN KGE training).
