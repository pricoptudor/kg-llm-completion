# Writeup notes — running log of findings

Insights worth putting in the technical report, tagged with the section they
likely belong to. Append as we go; curate at writing time (Week 4). Keep each
entry concrete enough that future-me doesn't have to re-derive it.

## Data / preprocessing

- **CRLF line-ending trap (FB15k-237 `.tsv`).** The triple files ship with
  Windows CRLF endings while the label files are LF-only. Naive line parsing
  leaves a trailing `\r` on every *tail* entity, so `/m/06cx9` and `/m/06cx9\r`
  are counted as different entities and the vocabulary silently doubles (27,395
  vs the true 14,541). Fix: strip `\r` before indexing and assert the canonical
  counts at load time. *(→ Method/reproducibility; a good "boring bugs that
  invalidate results" anecdote.)*

- **`entity2text.txt` is a superset.** KG-BERT's label file has 14,951 rows (the
  original FB15k count), not 14,541. The 410 extras are orphan entities left over
  from relations that FB15k-237 dropped. The ID space must be built from the
  triples, not the label file. *(→ Method: data.)*

## Evaluation method

- **Realistic tie-handling is not cosmetic.** Sparse or degenerate scorers
  produce huge blocks of tied scores (e.g. every frequency-0 entity). Optimistic
  ranking would hand such a model an undeservedly good rank; we use the
  optimistic/pessimistic average (PyKEEN's default). At scale it shows up as
  fractional median ranks (median 45.5 for the frequency baseline). *(→ Method:
  evaluation protocol.)*

- **Head + tail pooling.** Every test triple yields two queries; metrics are
  pooled over both (|Q| = 2·|test| = 40,932). Reporting one direction only is not
  comparable to published FB15k-237 numbers. *(→ Method: evaluation.)*

## Analysis / baselines

- **The frequency baseline is surprisingly strong on FB15k-237.** A
  relation-conditional "guess the most common object" model that **ignores the
  head entity entirely** scores filtered **MRR 0.2334 / H@1 0.1700 / H@3 0.2500 /
  H@10 0.3541** (test, head+tail pooled). This is dataset bias: many relations are
  heavily skewed toward a few objects (e.g. gender, language, nationality).
  Implications: (1) it is the honest floor for the results table — any KGE/LLM
  result must beat ~0.23 MRR by a real margin to mean anything; (2) the
  *per-relation* delta over this baseline is a more informative headline than the
  aggregate score. *(→ Analysis: per-relation breakdown; Limitations: dataset
  bias / contamination framing.)*

## KGE baselines (Phase 1, Day 3–5) — results and lessons

Final filtered MRR / Hits@k on FB15k-237 test (head+tail pooled, PyKEEN evaluator,
**dim 256**):

| Model   | MRR   | H@1   | H@3   | H@10  |
|---------|-------|-------|-------|-------|
| RotatE  | 0.324 | 0.229 | 0.361 | 0.518 |
| QuatE   | 0.304 | 0.217 | 0.335 | 0.483 |
| TransE  | 0.289 | 0.195 | 0.324 | 0.476 |
| ComplEx | 0.222 | 0.154 | 0.242 | 0.358 |

Lessons worth a paragraph each in the report:

- **1-vs-all (LCWA) sample efficiency, measured directly.** Same ComplEx model,
  dim, and epoch budget, three training regimes: LCWA+inverse **0.222**,
  LCWA-tail-only **0.138**, sLCWA(50 negs) **0.083**. LCWA scores all ~14.5k
  entities as implicit negatives per step vs sLCWA's 50, so it converges far faster
  at fixed epochs. Concrete evidence for a methods-section claim. *(→ Method.)*

- **Inverse triples are mandatory under LCWA — and they broke our scorer.** LCWA
  only trains tail prediction, so head-side MRR collapses (0.03) and pooled MRR
  halves. Adding inverse triples fixed the model (head 0.03→0.13, pooled
  0.138→0.222) but renumbered relations inside PyKEEN, so our manual scorer then
  read 0.005 while PyKEEN read 0.222. A *model improvement introduced a measurement
  bug*, caught only because we cross-check our harness against PyKEEN. Resolution:
  report KGE via PyKEEN; our harness (validated by exact agreement on a non-inverse
  model) is reserved for the LLM eval. *(→ Method / Limitations; good war story.)*

- **RotatE needs many steps; low LR looks like failure.** At lr 1e-4 / dim 256 /
  100 epochs RotatE scored 0.096 with validation MRR still climbing at the final
  epoch — undertrained, not broken. lr 5e-4 + 128 negs + 150 epochs → **0.324**.
  Reminder that a "bad" KGE number is often an optimization artifact. *(→ Method.)*

- **TransE beats ComplEx at dim 256.** TransE 0.289 vs ComplEx 0.222. ComplEx is
  dim-hungry (its strong published numbers use dim 1000–2000 + N3); at small dim a
  simple translation model wins. A nice "it depends" hook for the per-relation
  analysis. *(→ Analysis.)*
