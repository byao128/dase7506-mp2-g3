"""Freeze, verify, and officially evaluate the serialized G3 bundle."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
SOURCE_NAMES = (
    "common.py", "verifier.py", "features.py", "models.py",
    "group_fusion.py", "group_rank_features.py", "group_tree_ranker.py",
    "submission_group_tree.py",
)
ASSET_NAMES = (
    "MANIFEST.json", "inference.json", "pca.npz", "tree.json",
    "b0/model.json", "b0/weights.npz",
)
ASSET_LIMIT = 16 * 1024 * 1024


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_freeze(checkpoint, output):
    checkpoint = Path(checkpoint).resolve()
    actual_assets = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*") if path.is_file()
    }
    if actual_assets != set(ASSET_NAMES):
        difference = sorted(actual_assets.symmetric_difference(ASSET_NAMES))
        raise ValueError(f"Unexpected or missing G3 asset: {difference[0]}")
    paths = [CODE / name for name in SOURCE_NAMES]
    paths.extend(checkpoint / name for name in ASSET_NAMES)
    hashes = {
        path.relative_to(PROJECT).as_posix(): _digest(path) for path in paths
    }
    asset_bytes = sum((checkpoint / name).stat().st_size for name in ASSET_NAMES)
    if asset_bytes > ASSET_LIMIT:
        raise ValueError("G3 inference assets exceed 16 MiB")
    payload = {
        "format": "group-tree-v1-freeze",
        "checkpoint": checkpoint.relative_to(PROJECT).as_posix(),
        "method": "code/submission_group_tree.py",
        "asset_bytes": int(asset_bytes),
        "asset_limit_bytes": ASSET_LIMIT,
        "sha256": hashes,
    }
    Path(output).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def verify_freeze(checkpoint, freeze_path):
    checkpoint = Path(checkpoint).resolve()
    payload = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if payload.get("format") != "group-tree-v1-freeze":
        raise ValueError("Wrong G3 freeze format")
    if payload.get("checkpoint") != checkpoint.relative_to(PROJECT).as_posix():
        raise ValueError("Frozen checkpoint path does not match")
    actual_assets = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*") if path.is_file()
    }
    if actual_assets != set(ASSET_NAMES):
        difference = sorted(actual_assets.symmetric_difference(ASSET_NAMES))
        raise ValueError(f"Frozen asset set changed: {difference[0]}")
    expected_keys = {
        (CODE / name).relative_to(PROJECT).as_posix() for name in SOURCE_NAMES
    } | {
        (checkpoint / name).relative_to(PROJECT).as_posix()
        for name in ASSET_NAMES
    }
    hashes = payload.get("sha256", {})
    if set(hashes) != expected_keys:
        difference = sorted(set(hashes).symmetric_difference(expected_keys))
        raise ValueError(f"Frozen hash set changed: {difference[0]}")
    for key, expected in hashes.items():
        path = PROJECT / key
        if not path.is_file() or _digest(path) != expected:
            raise ValueError(f"Frozen file mismatch: {key}")
    asset_bytes = sum((checkpoint / name).stat().st_size for name in ASSET_NAMES)
    if asset_bytes != int(payload["asset_bytes"]) or asset_bytes > ASSET_LIMIT:
        raise ValueError("Frozen asset size changed or exceeds the limit")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--create-freeze", action="store_true")
    parser.add_argument("--split", choices=("validation", "test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=CODE / "data")
    args = parser.parse_args()
    if args.create_freeze:
        payload = create_freeze(args.checkpoint, args.freeze)
        print(json.dumps({key: value for key, value in payload.items()
                          if key != "sha256"}, indent=2))
        return
    if args.split is None or args.output is None:
        raise ValueError("evaluation requires --split and --output")
    verify_freeze(args.checkpoint, args.freeze)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "4"
    predictions = args.output.with_name(args.output.stem + "-predictions.json")
    predict_command = [
        sys.executable, str(CODE / "predict.py"),
        "--method", str(CODE / "submission_group_tree.py"),
        "--checkpoint", str(args.checkpoint), "--split", args.split,
        "--data-dir", str(args.data_dir), "--output", str(predictions),
    ]
    score_command = [
        sys.executable, str(CODE / "score_predictions.py"),
        "--split", args.split, "--data-dir", str(args.data_dir),
        "--predictions", str(predictions), "--output", str(args.output),
    ]
    started = time.perf_counter()
    subprocess.run(predict_command, check=True)
    subprocess.run(score_command, check=True)
    external_seconds = time.perf_counter() - started
    verify_freeze(args.checkpoint, args.freeze)
    result = json.loads(args.output.read_text(encoding="utf-8"))
    result.update({
        "external_seconds": external_seconds,
        "asset_bytes": json.loads(
            args.freeze.read_text(encoding="utf-8"))["asset_bytes"],
        "thread_environment": {
            name: os.environ[name] for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS")
        },
        "method": "code/submission_group_tree.py",
        "checkpoint": str(args.checkpoint),
        "freeze": str(args.freeze),
        "prediction_file": str(predictions),
    })
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
