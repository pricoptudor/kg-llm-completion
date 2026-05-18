"""Download FB15k-237 triples + KG-BERT entity/relation text mappings.

Idempotent: skips files that already exist with the right size.

Output layout (under data_cache/fb15k237/):
    train.tsv          head\trelation\ttail (Freebase MIDs / relation paths)
    valid.tsv          same format
    test.tsv           same format
    entity2text.txt    MID\thuman_name
    relation2text.txt  relation_path\thuman_name

Usage:
    python scripts/download_fb15k237.py
    python scripts/download_fb15k237.py --out-dir /kaggle/working/data_cache/fb15k237
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sources. We pull everything from the KG-BERT repo (yao8839836/kg-bert), which
# republishes the canonical Dettmers/ConvE FB15k-237 splits AND ships the
# entity/relation text mappings we need for the LLM side. Single source.
#
# Note: KG-BERT calls the validation split "dev.tsv" (NLP convention); we
# rename to valid.tsv on disk so the rest of the project uses the KG convention.
# ---------------------------------------------------------------------------

KGBERT_BASE = "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237"

# local_name -> remote_name
TRIPLE_FILES = {
    "train.tsv": "train.tsv",
    "valid.tsv": "dev.tsv",
    "test.tsv": "test.tsv",
}
LABEL_FILES = {
    "entity2text.txt": "entity2text.txt",
    "relation2text.txt": "relation2text.txt",
}

# Expected counts — used for a smoke check after download.
#
# Note on entity count: FB15k-237 has 14,541 entities (those still appearing in
# the 237 kept relations after Toutanova & Chen, 2015 pruned inverse-leakage
# relations from the original FB15k's 1,345). However, KG-BERT shipped the
# original FB15k entity2text.txt (14,951 entries) in their FB15k-237 folder.
# The extra 410 entities are orphans — they only appeared in relations that
# FB15k-237 removed. So entity2text.txt is a SUPERSET, not a corruption.
# We do a coverage check (every entity in the triples has a label) instead of
# a strict equality on file length.
EXPECTED_TRIPLE_COUNTS = {
    "train.tsv": 272_115,
    "valid.tsv": 17_535,
    "test.tsv": 20_466,
}
EXPECTED_ENTITY_COUNT = 14_541  # unique entities across train+valid+test
EXPECTED_RELATION_COUNT = 237   # unique relations across train+valid+test


def download(url: str, dest: Path, chunk_size: int = 1 << 14) -> None:
    """Stream a single URL to disk with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def file_hash(path: Path) -> str:
    """SHA256 of a file, for logging."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def count_lines(path: Path) -> int:
    """Count non-empty lines."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_triple_entities_relations(path: Path) -> tuple[set[str], set[str]]:
    """Return (entities, relations) appearing in a triples TSV (head\\trelation\\ttail)."""
    entities: set[str] = set()
    relations: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            h, r, t = parts
            entities.add(h)
            entities.add(t)
            relations.add(r)
    return entities, relations


def load_label_keys(path: Path) -> set[str]:
    """Return the set of keys (first tab-delimited field) in a label file."""
    keys: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            key = line.split("\t", 1)[0]
            keys.add(key)
    return keys


def fetch_if_missing(url: str, dest: Path) -> bool:
    """Download `url` to `dest` unless `dest` already has non-zero size.

    Returns True if a download happened, False if cached file was kept.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return False
    download(url, dest)
    return True


def smoke_check(out_dir: Path) -> None:
    """Verify the download is correct.

    Checks:
      1. Triple split sizes match published FB15k-237 counts (strict).
      2. Unique entity / relation counts across train+valid+test match (strict).
      3. Every entity and relation in the triples has a label (coverage check).
         The label files MAY be supersets — see note on EXPECTED_ENTITY_COUNT.
    """
    failures: list[str] = []

    # 1. Triple counts (strict).
    for fname, expected in EXPECTED_TRIPLE_COUNTS.items():
        actual = count_lines(out_dir / fname)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {fname:12s} {actual:>8,} lines (expected {expected:>8,})  [{status}]")
        if actual != expected:
            failures.append(f"{fname}: {actual} != {expected}")

    # 2. Unique entities/relations across all splits (strict).
    all_entities: set[str] = set()
    all_relations: set[str] = set()
    for fname in EXPECTED_TRIPLE_COUNTS:
        ents, rels = load_triple_entities_relations(out_dir / fname)
        all_entities |= ents
        all_relations |= rels

    print(
        f"  unique entities      {len(all_entities):>8,} "
        f"(expected {EXPECTED_ENTITY_COUNT:>8,})  "
        f"[{'OK' if len(all_entities) == EXPECTED_ENTITY_COUNT else 'MISMATCH'}]"
    )
    if len(all_entities) != EXPECTED_ENTITY_COUNT:
        failures.append(f"unique entities: {len(all_entities)} != {EXPECTED_ENTITY_COUNT}")

    print(
        f"  unique relations     {len(all_relations):>8,} "
        f"(expected {EXPECTED_RELATION_COUNT:>8,})  "
        f"[{'OK' if len(all_relations) == EXPECTED_RELATION_COUNT else 'MISMATCH'}]"
    )
    if len(all_relations) != EXPECTED_RELATION_COUNT:
        failures.append(f"unique relations: {len(all_relations)} != {EXPECTED_RELATION_COUNT}")

    # 3. Label coverage (label file may be a superset — that's fine).
    entity_labels = load_label_keys(out_dir / "entity2text.txt")
    relation_labels = load_label_keys(out_dir / "relation2text.txt")

    missing_ent = all_entities - entity_labels
    missing_rel = all_relations - relation_labels
    print(
        f"  entity2text          {len(entity_labels):>8,} keys "
        f"(missing {len(missing_ent)} of {len(all_entities)})  "
        f"[{'OK' if not missing_ent else 'MISSING'}]"
    )
    print(
        f"  relation2text        {len(relation_labels):>8,} keys "
        f"(missing {len(missing_rel)} of {len(all_relations)})  "
        f"[{'OK' if not missing_rel else 'MISSING'}]"
    )

    if missing_ent:
        sample = sorted(missing_ent)[:5]
        failures.append(
            f"entity2text missing {len(missing_ent)} entities (sample: {sample})"
        )
    if missing_rel:
        sample = sorted(missing_rel)[:5]
        failures.append(
            f"relation2text missing {len(missing_rel)} relations (sample: {sample})"
        )

    if failures:
        print("\nSmoke check FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FB15k-237 + text labels.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data_cache/fb15k237"),
        help="Where to put the files. Defaults to ./data_cache/fb15k237.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files exist.",
    )
    args = parser.parse_args()

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {out.resolve()}")

    # Everything comes from the KG-BERT repo now.
    all_files = {**TRIPLE_FILES, **LABEL_FILES}
    for local_name, remote_name in all_files.items():
        dest = out / local_name
        if args.force and dest.exists():
            dest.unlink()
        url = f"{KGBERT_BASE}/{remote_name}"
        action = "downloading" if fetch_if_missing(url, dest) else "cached"
        print(f"  {local_name:18s} {action:11s} sha256={file_hash(dest)}")

    print("\nSmoke check:")
    smoke_check(out)
    print("\nAll good.")


if __name__ == "__main__":
    main()
