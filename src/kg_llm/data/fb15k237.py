"""FB15k-237 data loading.

Loads the canonical FB15k-237 splits (KG-BERT republication) into integer-indexed
tensors plus the lookup maps we need for both KGE training and LLM prompting.

Design decisions (see CONTEXT_HANDOFF.md for the full list):

1. The entity/relation ID spaces are built from the *triples*, NOT from
   ``entity2text.txt``. That label file is a 14,951-entry superset (the original
   FB15k entity count); the 237 split actually uses only 14,541 entities. KGE
   embedding tables must be indexed over exactly the 14,541 entities that appear
   in the data, so the canonical ID space is the triple-derived one. Surplus
   labels are kept for name lookup but receive no ID.

2. IDs are assigned in **sorted order of the raw string keys**. This makes the
   mapping deterministic across machines, which matters because Tudor syncs the
   no-GPU laptop, the GTX 1650 box, and Kaggle through GitHub — a hash-ordered
   dict would otherwise reshuffle every embedding row between environments.

3. Lines are stripped of trailing ``\\r`` before parsing. The ``.tsv`` files in
   this repo are CRLF-terminated (they were written through Windows), while the
   ``.txt`` label files are LF-only. Without the strip, every *tail* entity
   carries a trailing carriage return and ``/m/06cx9`` vs ``/m/06cx9\\r`` get
   counted as two distinct entities — silently doubling the vocabulary.

4. The Freebase MID prefix ``/m/...`` is kept verbatim as the entity key.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

# Canonical FB15k-237 statistics. Used for an assertive sanity check at load
# time so a corrupted download can never silently poison every downstream run.
EXPECTED = {
    "num_entities": 14_541,
    "num_relations": 237,
    "train": 272_115,
    "valid": 17_535,
    "test": 20_466,
}

_SPLIT_FILES = {"train": "train.tsv", "valid": "valid.tsv", "test": "test.tsv"}
_ENTITY_TEXT = "entity2text.txt"
_RELATION_TEXT = "relation2text.txt"


def _read_triples(path: Path) -> list[tuple[str, str, str]]:
    """Read a tab-separated ``head<TAB>relation<TAB>tail`` file.

    ``line.strip()`` removes the trailing ``\\r\\n`` (CRLF, see module docstring)
    and any stray surrounding whitespace before we split on the tab. Entity MIDs
    and relation paths contain no internal whitespace, so this is safe.
    """
    triples: list[tuple[str, str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}: expected 3 tab-separated fields, got {parts!r}")
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _read_labels(path: Path) -> dict[str, str]:
    """Read a ``key<TAB>text`` label file into a dict.

    ``partition`` (not ``split``) so a label that itself contains a tab is kept
    intact in the text field rather than crashing the parse.
    """
    labels: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            key, _, text = line.partition("\t")
            labels[key] = text
    return labels


def _humanize_relation(relation: str) -> str:
    """Fallback readable name for a relation path with no entry in relation2text.

    e.g. ``/people/person/place_of_birth`` -> ``people person place of birth``.
    """
    return relation.strip("/").replace("/", " ").replace("_", " ").replace(".", " . ")


@dataclass(frozen=True)
class FilteredIndex:
    """Lookup tables of all *known-true* entities for filtered evaluation.

    Filtered evaluation (Bordes et al. 2013): when ranking the gold tail ``t`` of
    a test triple ``(h, r, t)`` against all entities, we must remove every *other*
    entity ``t'`` for which ``(h, r, t')`` is also a true triple anywhere in
    train+valid+test — otherwise a 1-to-N relation (e.g. Einstein's many awards)
    punishes the model for ranking another correct answer above the held-out one.

    These two dicts give, for any query, the set of true completions to exclude:
      - ``true_tails(h, r)``  -> all t with (h, r, t) known true
      - ``true_heads(r, t)``  -> all h with (h, r, t) known true

    Built from ALL splits combined — that union is what "filtered" means. The
    full ranking math lands in src/kg_llm/eval/ranking.py (Task 5); this is just
    the index it consumes.
    """

    hr_to_tails: dict[tuple[int, int], torch.Tensor]
    rt_to_heads: dict[tuple[int, int], torch.Tensor]

    def true_tails(self, head_id: int, relation_id: int) -> torch.Tensor:
        return self.hr_to_tails.get((head_id, relation_id), _EMPTY_LONG)

    def true_heads(self, relation_id: int, tail_id: int) -> torch.Tensor:
        return self.rt_to_heads.get((relation_id, tail_id), _EMPTY_LONG)


_EMPTY_LONG = torch.empty(0, dtype=torch.long)


@dataclass
class FB15k237:
    """In-memory FB15k-237 dataset: integer triples + string/name lookups."""

    train_triples: torch.Tensor  # (272115, 3) long: [head_id, relation_id, tail_id]
    valid_triples: torch.Tensor  # (17535, 3)
    test_triples: torch.Tensor  # (20466, 3)

    entity_to_id: dict[str, int]
    id_to_entity: dict[int, str]  # id -> Freebase MID, e.g. "/m/02mjmr"
    id_to_name: dict[int, str]  # id -> human label, e.g. "Barack Obama"

    relation_to_id: dict[str, int]
    id_to_relation: dict[int, str]  # id -> relation path
    id_to_relation_name: dict[int, str]  # id -> readable relation name

    @property
    def num_entities(self) -> int:
        return len(self.entity_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_to_id)

    def entity_name(self, entity_id: int) -> str:
        """Human label for an entity id, falling back to the MID if unlabeled."""
        return self.id_to_name.get(entity_id, self.id_to_entity[entity_id])

    def relation_name(self, relation_id: int) -> str:
        return self.id_to_relation_name[relation_id]

    def build_filtered_index(self) -> FilteredIndex:
        """Construct the filtered-eval index from train+valid+test combined."""
        all_triples = torch.cat([self.train_triples, self.valid_triples, self.test_triples], dim=0)

        hr_tails: dict[tuple[int, int], list[int]] = {}
        rt_heads: dict[tuple[int, int], list[int]] = {}
        for h, r, t in all_triples.tolist():
            hr_tails.setdefault((h, r), []).append(t)
            rt_heads.setdefault((r, t), []).append(h)

        hr_to_tails = {k: torch.tensor(sorted(set(v)), dtype=torch.long) for k, v in hr_tails.items()}
        rt_to_heads = {k: torch.tensor(sorted(set(v)), dtype=torch.long) for k, v in rt_heads.items()}
        return FilteredIndex(hr_to_tails=hr_to_tails, rt_to_heads=rt_to_heads)


def _encode(
    triples: list[tuple[str, str, str]],
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
) -> torch.Tensor:
    """Map a list of string triples to a (N, 3) LongTensor of ids."""
    rows = [
        (entity_to_id[h], relation_to_id[r], entity_to_id[t]) for h, r, t in triples
    ]
    return torch.tensor(rows, dtype=torch.long)


def load_fb15k237(data_dir: str | Path) -> FB15k237:
    """Load FB15k-237 from ``data_dir`` into a :class:`FB15k237`.

    ``data_dir`` must contain train.tsv / valid.tsv / test.tsv and the two label
    files entity2text.txt / relation2text.txt (the layout written by
    scripts/download_fb15k237.py into data_cache/fb15k237/).
    """
    data_dir = Path(data_dir)

    train = _read_triples(data_dir / _SPLIT_FILES["train"])
    valid = _read_triples(data_dir / _SPLIT_FILES["valid"])
    test = _read_triples(data_dir / _SPLIT_FILES["test"])

    # ID space comes from the triples only (see design note 1), assigned in
    # sorted string order for cross-machine determinism (design note 2).
    entity_keys = set()
    relation_keys = set()
    for split in (train, valid, test):
        for h, r, t in split:
            entity_keys.add(h)
            entity_keys.add(t)
            relation_keys.add(r)

    entity_to_id = {mid: i for i, mid in enumerate(sorted(entity_keys))}
    relation_to_id = {rel: i for i, rel in enumerate(sorted(relation_keys))}
    id_to_entity = {i: mid for mid, i in entity_to_id.items()}
    id_to_relation = {i: rel for rel, i in relation_to_id.items()}

    # Name lookups. entity2text may be a superset; we only keep labels for MIDs
    # that actually have an ID. relation2text may be missing a row or two — fall
    # back to a humanized version of the path so every relation has a name.
    entity_labels = _read_labels(data_dir / _ENTITY_TEXT)
    relation_labels = _read_labels(data_dir / _RELATION_TEXT)

    id_to_name = {
        i: entity_labels[mid] for i, mid in id_to_entity.items() if mid in entity_labels
    }
    id_to_relation_name = {
        i: relation_labels.get(rel) or _humanize_relation(rel)
        for i, rel in id_to_relation.items()
    }

    dataset = FB15k237(
        train_triples=_encode(train, entity_to_id, relation_to_id),
        valid_triples=_encode(valid, entity_to_id, relation_to_id),
        test_triples=_encode(test, entity_to_id, relation_to_id),
        entity_to_id=entity_to_id,
        id_to_entity=id_to_entity,
        id_to_name=id_to_name,
        relation_to_id=relation_to_id,
        id_to_relation=id_to_relation,
        id_to_relation_name=id_to_relation_name,
    )

    _validate(dataset)
    return dataset


def _validate(ds: FB15k237) -> None:
    """Assert the canonical FB15k-237 counts so a bad download fails loudly."""
    assert ds.num_entities == EXPECTED["num_entities"], (
        f"entity count {ds.num_entities} != {EXPECTED['num_entities']}; "
        "check for stray \\r (CRLF) or a wrong download source."
    )
    assert ds.num_relations == EXPECTED["num_relations"], (
        f"relation count {ds.num_relations} != {EXPECTED['num_relations']}"
    )
    for split in ("train", "valid", "test"):
        n = getattr(ds, f"{split}_triples").shape[0]
        assert n == EXPECTED[split], f"{split} has {n} triples, expected {EXPECTED[split]}"
