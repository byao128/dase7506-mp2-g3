"""Verify and evaluate the supplied G3 checkpoint in two fresh processes."""

import os

for _name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from evaluate_group_tree_bundle import verify_freeze


CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
DEFAULT_CHECKPOINT = PROJECT / "artifacts/group-tree-v1/bundle"
DEFAULT_FREEZE = PROJECT / "records/group-tree-v1/freeze.json"
THREAD_NAMES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_process(command, stdout_path, stderr_path, env):
    started_unix = time.time()
    started = time.perf_counter()
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    result = {
        "pid": process.pid,
        "returncode": process.returncode,
        "started_unix": started_unix,
        "wall_seconds": elapsed,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
    }
    if process.returncode:
        message = stderr.decode("utf-8", errors="replace")
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=message)
    return result


def evaluate(data_dir, checkpoint, freeze, split, output):
    """Run official predict.py and score_predictions.py without rewriting score."""
    data_dir = Path(data_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    freeze = Path(freeze).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")

    frozen = verify_freeze(checkpoint, freeze)
    output.mkdir(parents=True)
    predictions = output / "predictions.json"
    raw_score = output / "raw-score.json"
    env = os.environ.copy()
    env.update({name: "4" for name in THREAD_NAMES})
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(CODE) if not existing_pythonpath
        else str(CODE) + os.pathsep + existing_pythonpath)

    predict_command = [
        sys.executable, str(CODE / "predict.py"),
        "--method", str(CODE / "submission_group_tree.py"),
        "--checkpoint", str(checkpoint),
        "--split", split,
        "--data-dir", str(data_dir),
        "--output", str(predictions),
    ]
    score_command = [
        sys.executable, str(CODE / "score_predictions.py"),
        "--predictions", str(predictions),
        "--split", split,
        "--data-dir", str(data_dir),
        "--output", str(raw_score),
    ]
    overall_started_unix = time.time()
    overall_started = time.perf_counter()
    processes = {}
    try:
        processes["predict"] = _run_process(
            predict_command, output / "predict.stdout.log",
            output / "predict.stderr.log", env)
        processes["score"] = _run_process(
            score_command, output / "score.stdout.log",
            output / "score.stderr.log", env)
        verify_freeze(checkpoint, freeze)
    except Exception:
        failure = {
            "schema": 1,
            "status": "failed",
            "split": split,
            "started_unix": overall_started_unix,
            "external_wall_seconds": time.perf_counter() - overall_started,
            "processes": processes,
            "thread_environment": {name: env[name] for name in THREAD_NAMES},
        }
        (output / "resources.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        raise

    resources = {
        "schema": 1,
        "status": "completed",
        "purpose": "fixed-checkpoint-reproduction; not method selection",
        "split": split,
        "started_unix": overall_started_unix,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "external_wall_seconds": time.perf_counter() - overall_started,
        "processes": processes,
        "thread_environment": {name: env[name] for name in THREAD_NAMES},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "catboost")
        },
        "checkpoint_asset_bytes": frozen["asset_bytes"],
        "freeze_sha256": _sha256(freeze),
        "data_manifest_sha256": _sha256(data_dir / "manifest.json"),
        "files": {
            "predictions": predictions.name,
            "raw_score": raw_score.name,
        },
    }
    (output / "resources.json").write_text(
        json.dumps(resources, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return resources


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the fixed G3 score with fresh official processes.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--split", choices=("train", "validation", "test", "audit"),
                        default="test")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; it must not exist")
    args = parser.parse_args()
    resources = evaluate(
        args.data_dir, args.checkpoint, args.freeze, args.split, args.output)
    print(json.dumps(resources, indent=2), flush=True)


if __name__ == "__main__":
    main()
