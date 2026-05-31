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
