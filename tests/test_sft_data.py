"""Formatting checks for SFT data — pure Python, no torch/datasets needed."""

from kg_llm.llm.sft_data import head_question, tail_question, triple_to_examples


def test_both_directions():
    ex = triple_to_examples("Albert Einstein", "place of birth", "Ulm")
    assert len(ex) == 2

    # Tail prediction: ask for the tail given the head; answer is the tail.
    tail = ex[0]["messages"]
    assert tail[0]["role"] == "user"
    assert "Albert Einstein" in tail[0]["content"]
    assert "place of birth" in tail[0]["content"]
    assert tail[1] == {"role": "assistant", "content": "Ulm"}

    # Head prediction: ask for the head given the tail; answer is the head.
    head = ex[1]["messages"]
    assert head[0]["role"] == "user"
    assert "Ulm" in head[0]["content"]
    assert head[1] == {"role": "assistant", "content": "Albert Einstein"}


def test_question_wording():
    assert "tail entity" in tail_question("H", "R")
    assert "head entity" in head_question("T", "R")
    assert "entity name only" in tail_question("H", "R")
