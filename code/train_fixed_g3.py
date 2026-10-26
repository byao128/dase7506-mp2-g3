"""Train the frozen G3 recipe on official train data at an explicit path."""

import os

for _name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import check_data, load_inputs, parse_answer, read_jsonl
from group_tree_bundle import fit_and_save_bundle


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
PROVIDED_CHECKPOINT = (PROJECT / "artifacts/group-tree-v1/bundle").resolve()
FIXED_CONFIG = {
    "depth": 4,
    "l2_leaf_reg": 10.0,
    "tree_count": 600,
    "alpha": 0.5,
}
FIXED_SEEDS = {"b0": 2028, "b0_oof_split": 6100, "tree": 2028}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_train_supervision(data_dir):
    """Load official train labels while honoring an explicit data directory."""
    data_dir = Path(data_dir).resolve()
    check_data(data_dir, ("train",))
    bank, arrays = load_inputs("train", data_dir)
    references = read_jsonl(data_dir / "gsm8k_train.jsonl")
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
            labels[row_index, candidate_index] = (
                answer is not None and answer == gold)
    embeddings = np.asarray(arrays["embeddings"])
    if embeddings.shape[:2] != (len(bank), 4):
        raise ValueError("Expected embeddings with shape [N, 4, D]")
    return bank, embeddings, labels


def train_fixed(data_dir, output, record):
    """Fit one fixed G3 bundle without selecting hyperparameters."""
    data_dir = Path(data_dir).resolve()
    output = Path(output).resolve()
    record = Path(record).resolve()
    if output == PROVIDED_CHECKPOINT:
        raise FileExistsError("The provided checkpoint is immutable")
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    if record.exists():
        raise FileExistsError(f"Training record already exists: {record}")

    bank, embeddings, labels = load_train_supervision(data_dir)
    manifest = fit_and_save_bundle(
        bank, embeddings, labels, dict(FIXED_CONFIG), output)
    payload = {
        **manifest,
        "schema": 1,
        "recipe": "fixed-original-g3",
        "config": dict(FIXED_CONFIG),
        "seeds": dict(FIXED_SEEDS),
        "training_rows": len(bank),
        "train_only": True,
        "validation_accessed": False,
        "test_accessed": False,
        "hyperparameters_reselected": False,
        "data_manifest_sha256": _sha256(data_dir / "manifest.json"),
        "thread_environment": {
            name: os.environ[name] for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")
        },
    }
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Train the fixed original G3 recipe on official train data.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New checkpoint directory; it must not exist")
    parser.add_argument("--record", type=Path, required=True,
                        help="New JSON training record; it must not exist")
    args = parser.parse_args()
    payload = train_fixed(args.data_dir, args.output, args.record)
    print(json.dumps({
        "format": payload["format"],
        "config": payload["config"],
        "training_rows": payload["training_rows"],
        "fit_seconds": payload["fit_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
