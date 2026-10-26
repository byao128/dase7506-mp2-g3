"""Command-line entry point for the registered G3 train-only experiment."""

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from group_tree_cv import run_group_tree_oof
from supervision import load_supervised


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-seed", type=int, default=2026)
    parser.add_argument("--output", default="../results/group-tree-v1")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "4"
    bank, embeddings, labels, hard_mask = load_supervised("train")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema": 1, "experiment": "G3", "fold_seed": args.fold_seed,
        "train_rows": len(bank), "hard_rows": int(hard_mask.sum()),
        "feature_count": 1438, "validation_accessed": False,
        "test_accessed": False, "started_unix": time.time(),
        "python": platform.python_version(), "numpy": np.__version__,
    }
    (output / f"preflight-seed-{args.fold_seed}.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def observer(record):
        print(json.dumps(record, ensure_ascii=False), flush=True)

    summary = run_group_tree_oof(
        bank, embeddings, labels, hard_mask, fold_seed=args.fold_seed,
        output_dir=output, observer=observer)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in {"b0_choices", "g3_choices", "folds"}},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
