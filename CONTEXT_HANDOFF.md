# Conversation handoff — KG_LLM_Completion

This document is a checkpoint for any future Claude conversation picking up
work on this project. The authoritative plan is `kg_llm_project_plan.md`;
this file captures **where we are inside that plan** as of the most recent
session, plus the working agreements that aren't visible from code.

> Last updated: 2026-05-31 (session that completed Week 1 Day 1–2 — data loader,
> filtered-eval harness, and the frequency-baseline smoke test, all green).

---

## Working agreement (do this every session)

- **Pair-programming mode.** Claude writes code directly into the workspace
  folder with step-by-step explanations. Tudor reviews and makes architectural
  changes / pushes back when warranted.
- **Theory depth: intuition + key equations — but explain from the ground up.**
  Not full derivations, no hand-waving. CRITICAL (2026-05-31 feedback): do NOT
  drop unexplained jargon or bare symbols. Define every term/symbol the first
  time it appears, lead with intuition and a concrete running example, and only
  then show the equation. Tudor is strong at math but new to KGE/LLM-training
  vocabulary — a dense block forced him to look terms up separately, which
  defeats the learning goal. Reference the relevant paper.
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

We have **completed Phase 1, Week 1, Day 1–2** of the 8-week plan and are
starting **Day 3–5 (PyKEEN KGE baseline training)**.

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
- [x] **Data loader** `src/kg_llm/data/fb15k237.py`: `load_fb15k237()` returns a
      `FB15k237` dataclass with train/valid/test as `(N,3)` LongTensors plus all
      id↔MID↔name maps; `build_filtered_index()` returns a `FilteredIndex` of
      known-true heads/tails. Asserts canonical counts at load. Verified clean
      (14,541 / 237 / 272,115 / 17,535 / 20,466).
- [x] **Filtered-eval harness** `src/kg_llm/eval/ranking.py`: `filtered_ranks`
      (realistic tie handling), `aggregate` (MRR + Hits@{1,3,10}), `evaluate`
      (pools head+tail). Brute-force toy test in `tests/test_ranking.py` — green.
- [x] **End-to-end smoke** `src/kg_llm/eval/baselines.py` (`FrequencyBaseline`)
      + `scripts/run_frequency_baseline.py`. Filtered MRR 0.2334 / H@1 0.1700 /
      H@3 0.2500 / H@10 0.3541 on test; matches an independent NumPy recompute.

### Open tasks

All Day 1–2 tasks are complete. The TaskCreate list at this handoff:

| # | Status        | Title                                                | Notes |
|---|---------------|------------------------------------------------------|-------|
| 1 | completed     | Conceptual foundations: KGs, link prediction, FB15k-237 | |
| 2 | completed     | Write repo scaffolding + git init                    | |
| 3 | completed     | Download FB15k-237 and inspect                       | Coverage smoke check passing. |
| 4 | completed     | Build data loading module                            | `src/kg_llm/data/fb15k237.py`. |
| 5 | completed     | Filtered evaluation: theory + implementation         | `src/kg_llm/eval/ranking.py` + `tests/test_ranking.py`. |
| 6 | completed     | Verify Day 1–2 setup end-to-end                      | Frequency baseline; harness produces sane numbers. |
| 7 | next          | KGE baselines (Day 3–5)                              | PyKEEN ComplEx/RotatE/TransE/QuatE on FB15k-237; wrap as `Scorer`s; save embeddings for hard-negative mining. |

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
8. **ID space built from the triples, in sorted string order** — not from the
   label files, not hash order. Sorted order makes the id↔entity mapping
   deterministic across Tudor's machines, so embedding rows don't reshuffle on
   GitHub sync. The label superset (entity2text) gets no ID of its own.
9. **`Scorer` protocol** (`score_tails`/`score_heads` → `(batch, num_entities)`)
   is the single interface every model implements: frequency baseline, PyKEEN
   KGE, and the LLM log-prob scorer. The eval harness never special-cases a model.
10. **Realistic tie handling** (optimistic/pessimistic average, PyKEEN default)
    in filtered ranking — chosen so degenerate constant scorers can't game Hits@k.
11. **`FilteredIndex` is opt-in** (`ds.build_filtered_index()`), not baked into
    `load()`: it materializes dicts over all ~310k triples and not every caller
    needs it.
12. **Trivial baselines live in `src/kg_llm/eval/baselines.py`** for now (their
    job is harness validation). May move next to the real models later.

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
  (Corollary: the sandbox also can't `pip install torch`, so torch-dependent
  code is verified there via a tiny torch stub or an independent NumPy mirror,
  and the real run happens on Tudor's machine.)

- **CRLF line-ending trap.** The `.tsv` triple files are CRLF-terminated (written
  through Windows); the `.txt` label files are LF-only. Without stripping `\r`,
  every tail entity carries a trailing carriage return and the entity vocab
  silently doubles (27,395 vs the true 14,541). The loader strips `\r` and
  asserts canonical counts so this can't slip through again.

- **The frequency baseline is surprisingly strong on FB15k-237** (filtered MRR
  ≈ 0.23, ignoring the head entity entirely). Dataset bias — many relations are
  skewed toward a few objects. It's the honest floor for the results table;
  per-relation deltas over it are more informative than the aggregate.

- **torch DLL hell on Windows.** A CUDA build of torch on a no-GPU laptop (and/or
  an Anaconda-derived venv pulling conflicting MKL/OpenMP DLLs) throws
  `WinError 1114 ... c10.dll`. Fix that stuck: rebuild the venv with the
  python.org `py` launcher (not Anaconda) and install the CPU torch wheel.

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
- The *math* of filtered evaluation (covered this session): rank = 1 + #strictly
  greater after filtering; realistic ties add 0.5 each; MRR = mean(1/rank);
  Hits@k = mean(rank ≤ k); every test triple yields a head AND a tail query and
  the two are pooled (|Q| = 2·|test|).

## Conceptual context still to cover (queued)

- KGE training loss (negative sampling, NSSA — Sun et al. 2019); margin loss
  vs cross-entropy with negatives. (Day 3–5, coming next.)
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
| `src/kg_llm/data/fb15k237.py`         | Data loader + `FilteredIndex`.                     |
| `src/kg_llm/eval/ranking.py`          | Filtered ranking harness (`Scorer`, `evaluate`).   |
| `src/kg_llm/eval/baselines.py`        | `FrequencyBaseline` (harness sanity floor).        |
| `tests/test_ranking.py`               | Brute-force toy verification of the harness.       |
| `scripts/run_frequency_baseline.py`   | End-to-end Day 1–2 smoke entry point.              |
| `reports/writeup_notes.md`            | Running log of findings for the technical report.  |
| `CONTEXT_HANDOFF.md`                  | This file.                                         |

---

## Next concrete action when picking up

Day 1–2 is done. We are moving into **Week 1, Day 3–5: KGE baselines**.

1. Re-read this file and `kg_llm_project_plan.md` Week 1 Day 3–5.
2. Cover the theory first (Tudor's standing request — intuition + key equations):
   KGE training loss, negative sampling, self-adversarial negative sampling
   (NSSA, Sun et al. 2019), margin vs cross-entropy objectives.
3. Train **ComplEx, RotatE, TransE, QuatE** on FB15k-237 with PyKEEN, standard
   hyperparameters. Heavy training runs on Kaggle T4 (sync via GitHub), not the
   laptop. Sanity target: ComplEx ≈ 0.32 MRR, RotatE ≈ 0.34 filtered.
4. Wrap each trained model as a `Scorer` (see decision #9) so it drops straight
   into `evaluate()`; cross-check our harness numbers against PyKEEN's own
   evaluator on at least one model (must agree — validates both).
5. **Save the trained embeddings** — they're the raw material for Week 3's
   KGE-mined hard negatives (the project's original angle).

Reminder of the working agreement: pause after meaningful units for Tudor's
review; he runs all git/pip/training on his side; explain the theory as we build.
