"""Training and validation supervision built with the official answer parser."""

from pathlib import Path

import numpy as np

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import answer_metadata


def load_supervised(split: str):
    """Load an allowed split and retain every candidate matching the reference."""
    if split not in {"train", "validation"}:
        raise ValueError("Supervision is available only for train and validation")

    bank, arrays = load_inputs(split)
    references = read_jsonl(Path(DATA) / f"gsm8k_{split}.jsonl")
    if len(references) != len(bank):
        raise ValueError("Reference and candidate bank lengths differ")

    labels = np.zeros((len(bank), 4), dtype=np.bool_)
    for row_index, (row, reference) in enumerate(zip(bank, references)):
        if (reference.get("id") != row.get("id") or
                reference.get("question") != row.get("question")):
            raise ValueError("Reference and candidate bank alignment mismatch")
        gold = parse_answer(reference.get("answer", ""))
        if gold is None:
            raise ValueError("Reference answer is not parseable")
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            labels[row_index, candidate_index] = answer is not None and answer == gold

    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape[:2] != (len(bank), 4):
        raise ValueError("Expected embeddings with shape [N, 4, D]")
    hard_mask = answer_metadata(bank)["hard_mask"]
    return bank, embeddings, labels, hard_mask
