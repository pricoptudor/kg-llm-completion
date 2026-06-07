"""Format FB15k-237 triples as chat-style SFT examples (both directions).

Each triple (h, r, t) yields TWO training examples — predict the tail, predict the
head — as chat messages (a user question + the assistant's answer). The training
script applies the model's chat template and masks the loss to the assistant answer
only (Week 2 notes). Questions use the human-readable entity/relation names.

IMPORTANT: the question wording here is the single source of truth. The SFT-model
evaluation must score candidates with the SAME wording + chat template, or we get a
train/eval mismatch. (`tail_question`/`head_question` are reused by the scorer's
chat mode.)

This module deliberately keeps no heavy top-level imports (no torch/datasets), so
the pure formatting helpers are trivially testable; `datasets` is imported lazily.
"""

from __future__ import annotations

_ANSWER_HINT = "Answer with the entity name only."


def tail_question(head_name: str, relation_name: str) -> str:
    return (
        f"Given the head entity '{head_name}' and the relation '{relation_name}', "
        f"what is the tail entity? {_ANSWER_HINT}"
    )


def head_question(tail_name: str, relation_name: str) -> str:
    return (
        f"Given the tail entity '{tail_name}' and the relation '{relation_name}', "
        f"what is the head entity? {_ANSWER_HINT}"
    )


def triple_to_examples(head_name: str, relation_name: str, tail_name: str) -> list[dict]:
    """Two chat examples for one triple: tail prediction, then head prediction."""
    return [
        {
            "messages": [
                {"role": "user", "content": tail_question(head_name, relation_name)},
                {"role": "assistant", "content": tail_name},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": head_question(tail_name, relation_name)},
                {"role": "assistant", "content": head_name},
            ]
        },
    ]


def make_sft_dataset(ds, split: str = "train", max_triples: int | None = None):
    """Build a HuggingFace `Dataset` of chat examples for the given split.

    One column, `messages` (a list of {role, content}); two rows per triple.
    `max_triples` caps the number of source triples — handy for a quick smoke run.
    """
    from datasets import Dataset  # lazy import: keep the module light to import

    triples = getattr(ds, f"{split}_triples").tolist()
    if max_triples is not None:
        triples = triples[:max_triples]

    def gen():
        for h, r, t in triples:
            yield from triple_to_examples(
                ds.entity_name(h), ds.relation_name(r), ds.entity_name(t)
            )

    return Dataset.from_generator(gen)
