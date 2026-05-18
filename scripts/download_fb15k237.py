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
# Sources. We pull from raw.githubusercontent so the URLs are stable.
# ---------------------------------------------------------------------------

# Triples (head\trelation\ttail) — from the ConvE repo, the canonical FB15k-237 split.
TRIPLES_BASE = "https://raw.githubusercontent.com/TimDettmers/ConvE/master/FB15k-237"
TRIPLE_FILES = {
    "train.tsv": "train.txt",
    "valid.tsv": "valid.txt",
    "test.tsv": "test.txt",
}

# Entity/relation text labels — from KG-BERT.
KGBERT_BASE = "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237"
LABEL_FILES = {
    "entity2text.txt": "entity2text.txt",
    "relation2text.txt": "relation2text.txt",
}

# Expected counts — used for a smoke check after download.
EXPECTED_TRIPLE_COUNTS = {
    "train.tsv": 272_115,
    "valid.tsv": 17_535,
    "test.tsv": 20_466,
}
EXPECTED_ENTITY_COUNT = 14_541
EXPECTED_RELATION_COUNT = 237


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


def fetch_if_missing(url: str, dest: Path) -> bool:
    """Download `url` to `dest` unless `dest` already has non-zero size.

    Returns True if a download happened, False if cached file was kept.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return False
    download(url, dest)
    return True


def smoke_check(out_dir: Path) -> None:
    """Verify line counts match the published splits — catch a corrupted download."""
    failures: list[str] = []

    for fname, expected in EXPECTED_TRIPLE_COUNTS.items():
        actual = count_lines(out_dir / fname)
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {fname:12s} {actual:>8,} lines (expected {expected:>8,})  [{status}]")
        if actual != expected:
            failures.append(f"{fname}: {actual} != {expected}")

    n_ent = count_lines(out_dir / "entity2text.txt")
    print(f"  entity2text  {n_ent:>8,} entries (expected {EXPECTED_ENTITY_COUNT:>8,})")
    if n_ent != EXPECTED_ENTITY_COUNT:
        failures.append(f"entity2text: {n_ent} != {EXPECTED_ENTITY_COUNT}")

    n_rel = count_lines(out_dir / "relation2text.txt")
    print(f"  relation2text {n_rel:>7,} entries (expected {EXPECTED_RELATION_COUNT:>8,})")
    if n_rel != EXPECTED_RELATION_COUNT:
        failures.append(f"relation2text: {n_rel} != {EXPECTED_RELATION_COUNT}")

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

    # Triples
    for local_name, remote_name in TRIPLE_FILES.items():
        dest = out / local_name
        if args.force and dest.exists():
            dest.unlink()
        url = f"{TRIPLES_BASE}/{remote_name}"
        action = "downloading" if fetch_if_missing(url, dest) else "cached"
        print(f"  {local_name:18s} {action:11s} sha256={file_hash(dest)}")

    # Entity / relation labels
    for local_name, remote_name in LABEL_FILES.items():
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
