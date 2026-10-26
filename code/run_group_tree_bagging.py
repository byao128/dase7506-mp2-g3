"""Restricted CLI for the fixed G3 query-bagging train-only experiment."""

import os

_THREAD_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _thread_key in _THREAD_KEYS:
    os.environ[_thread_key] = "4"

import argparse
import builtins
import hashlib
import io
import json
from pathlib import Path
import platform
import time
import traceback
import uuid

import numpy as np


DEFAULT_OUTPUT = "../results/group-tree-bagging-v1"
TRAIN_DATA_RELATIVE = (
    "code/data/manifest.json",
    "code/data/candidates/gsm8k_train.jsonl",
    "code/data/candidates/gsm8k_train.npz",
    "code/data/gsm8k_train.jsonl",
)
G3_FREEZE_RELATIVE = "records/group-tree-v1/freeze.json"
G3_BINDING_RELATIVE = (
    "artifacts/group-tree-v1/bundle/MANIFEST.json",
    "artifacts/group-tree-v1/bundle/b0/model.json",
    "artifacts/group-tree-v1/bundle/b0/weights.npz",
    "artifacts/group-tree-v1/bundle/inference.json",
    "artifacts/group-tree-v1/bundle/pca.npz",
    "artifacts/group-tree-v1/bundle/tree.json",
    "code/common.py",
    "code/features.py",
    "code/group_fusion.py",
    "code/group_rank_features.py",
    "code/group_tree_ranker.py",
    "code/models.py",
    "code/submission_group_tree.py",
    "code/verifier.py",
)


def _resolved(value):
    if isinstance(value, int):
        return None
    try:
        return Path(value).resolve()
    except (TypeError, OSError, ValueError):
        return None


def _is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validated_output(project_root, value):
    """Resolve only the registered new result namespace."""
    root = Path(project_root).resolve()
    expected = (root / "results" / "group-tree-bagging-v1").resolve()
    output = Path(value).resolve()
    if output != expected:
        raise ValueError(
            f"output must be the fixed experiment namespace: {expected}")
    return output


class ReadAudit:
    """Record reads and enforce the train-only experiment namespace."""

    def __init__(self, project_root, output_root):
        self.root = Path(project_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.data_root = (self.root / "code" / "data").resolve()
        self.results_root = (self.root / "results").resolve()
        self.artifacts_root = (self.root / "artifacts").resolve()
        self.allowed_train = {
            (self.root / value).resolve() for value in TRAIN_DATA_RELATIVE
        }
        self.allowed_g3 = {
            (self.root / value).resolve()
            for value in (G3_FREEZE_RELATIVE, *G3_BINDING_RELATIVE)
        }
        self.read_paths = set()
        self.write_paths = set()
        self._builtins_open = None
        self._io_open = None

    def _observe(self, value, mode):
        path = _resolved(value)
        if path is None:
            return
        mode = str(mode)
        reading = "r" in mode or "+" in mode
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        in_output = _is_within(path, self.output_root)
        if _is_within(path, self.data_root):
            if writing or (reading and path not in self.allowed_train):
                raise PermissionError(f"forbidden experiment data access: {path}")
        if _is_within(path, self.results_root) and not in_output:
            if writing or (reading and path not in self.allowed_g3):
                raise PermissionError(f"forbidden prior-result access: {path}")
        if _is_within(path, self.artifacts_root) and not in_output:
            if writing or (reading and path not in self.allowed_g3):
                raise PermissionError(f"forbidden artifact access: {path}")
        if reading:
            self.read_paths.add(str(path))
        if writing:
            self.write_paths.add(str(path))

    def __enter__(self):
        self._builtins_open = builtins.open
        self._io_open = io.open

        def audited_builtin(file, mode="r", *args, **kwargs):
            self._observe(file, mode)
            return self._builtins_open(file, mode, *args, **kwargs)

        def audited_io(file, mode="r", *args, **kwargs):
            self._observe(file, mode)
            return self._io_open(file, mode, *args, **kwargs)

        builtins.open = audited_builtin
        io.open = audited_io
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        builtins.open = self._builtins_open
        io.open = self._io_open

    def observed_data_reads(self):
        return sorted(str(path) for path in self.allowed_train
                      if str(path) in self.read_paths)


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official_train_hashes(data_dir):
    data_dir = Path(data_dir)
    relative = (
        "manifest.json", "candidates/gsm8k_train.jsonl",
        "candidates/gsm8k_train.npz", "gsm8k_train.jsonl",
    )
    return {name: _sha256_file(data_dir / name) for name in relative}


def verify_g3_freeze(project_root):
    """Verify every one of the incumbent freeze's fourteen bindings."""
    root = Path(project_root).resolve()
    freeze_path = root / G3_FREEZE_RELATIVE
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected = freeze.get("sha256")
    if (freeze.get("format") != "group-tree-v1-freeze" or
            not isinstance(expected, dict) or len(expected) != 14 or
            set(expected) != set(G3_BINDING_RELATIVE)):
        raise ValueError("incumbent G3 freeze binding set is invalid")
    actual = {name: _sha256_file(root / name) for name in sorted(expected)}
    matches = {name: actual[name] == expected[name] for name in sorted(expected)}
    if not all(matches.values()):
        changed = [name for name, matches_expected in matches.items()
                   if not matches_expected]
        raise RuntimeError(f"incumbent G3 binding changed: {changed}")
    return {
        "freeze_path": G3_FREEZE_RELATIVE,
        "freeze_sha256": _sha256_file(freeze_path),
        "binding_count": 14,
        "bindings": actual,
        "all_match": True,
    }


def load_official_train():
    from common import check_data
    from supervision import load_supervised
    check_data(splits=("train",))
    return load_supervised("train")


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8")


def publish_result_atomic(path, payload):
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite completed phase: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    try:
        _write_json(temporary, payload)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite completed phase: {target}")
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return target


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("primary", "stability"))
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser


def _runtime_versions():
    try:
        import catboost
        catboost_version = catboost.__version__
    except (ImportError, AttributeError):
        catboost_version = "unavailable"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "catboost": catboost_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {key: os.environ[key] for key in _THREAD_KEYS},
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    code_dir = Path(__file__).resolve().parent
    root = code_dir.parent
    output = validated_output(root, args.output)
    phase_dir = output / args.stage
    result_path = phase_dir / f"{args.stage}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed {args.stage} phase")

    attempt_id = f"{int(time.time())}-{uuid.uuid4().hex}"
    attempt_dir = phase_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=False)
    progress_path = attempt_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    started = time.perf_counter()
    runtime = _runtime_versions()

    with ReadAudit(root, output) as audit:
        try:
            from common import DATA
            from crossfit import canonical_sha256
            from group_tree_bagging_cv import (
                FIXED_CONFIG, _source_hashes, run_bagging_oof)

            source_before = _source_hashes()
            train_before = official_train_hashes(DATA)
            g3_before = verify_g3_freeze(root)
            primary = None
            if args.stage == "stability":
                primary_path = output / "primary" / "primary.json"
                if not primary_path.is_file():
                    raise FileNotFoundError(
                        "stability requires completed primary result")
                primary = json.loads(primary_path.read_text(encoding="utf-8"))
                if primary.get("gate_passed") is not True:
                    raise ValueError("primary gate did not pass; stability is prohibited")
                if primary.get("source_hashes") != source_before:
                    raise ValueError("primary source binding does not match stability")
                if primary.get("config_sha256") != canonical_sha256(FIXED_CONFIG):
                    raise ValueError("primary fixed config does not match stability")
                if primary.get("official_train_hashes") != train_before:
                    raise ValueError("primary train files do not match stability")

            bank, embeddings, labels, loader_hard_mask = load_official_train()
            fold_seed = FIXED_CONFIG["fold_seeds"][args.stage]
            preflight = {
                "experiment": "G3-query-bagging",
                "schema": 1,
                "stage": args.stage,
                "fold_seed": fold_seed,
                "train_rows": int(len(bank)),
                "loader_hard_rows": int(np.asarray(loader_hard_mask, dtype=bool).sum()),
                "fixed_config": FIXED_CONFIG,
                "fixed_config_sha256": canonical_sha256(FIXED_CONFIG),
                "source_hashes": source_before,
                "official_train_hashes": train_before,
                "incumbent_g3": g3_before,
                "runtime": runtime,
                "validation_accessed": False,
                "test_accessed": False,
                "started_unix": time.time(),
                "attempt_id": attempt_id,
            }
            _write_json(attempt_dir / "preflight.json", preflight)

            def observer(record):
                with progress_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(
                        record, ensure_ascii=False, sort_keys=True) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)

            result = run_bagging_oof(
                bank, embeddings, labels, fold_seed=fold_seed,
                output_dir=output, observer=observer)
            if not np.array_equal(
                    np.asarray(result["hard_mask"], dtype=bool),
                    np.asarray(loader_hard_mask, dtype=bool)):
                raise RuntimeError("official loader hard mask changed during CV")

            source_after = _source_hashes()
            train_after = official_train_hashes(DATA)
            g3_after = verify_g3_freeze(root)
            if source_after != source_before:
                raise RuntimeError("source files changed during the phase")
            if train_after != train_before:
                raise RuntimeError("official train files changed during the phase")
            if g3_after != g3_before:
                raise RuntimeError("incumbent G3 changed during the phase")
            result["official_train_hashes"] = train_before
            result["observed_reads"] = sorted(audit.read_paths)
            result["observed_data_reads"] = audit.observed_data_reads()
            result["integrity"] = {
                "source_before": source_before,
                "source_after": source_after,
                "source_unchanged": True,
                "official_train_before": train_before,
                "official_train_after": train_after,
                "official_train_unchanged": True,
                "incumbent_g3_before": g3_before,
                "incumbent_g3_after": g3_after,
                "incumbent_g3_unchanged": True,
            }
            result["runtime"] = {
                **runtime,
                "cli_wall_seconds": float(time.perf_counter() - started),
                "attempt_id": attempt_id,
            }
            publish_result_atomic(result_path, result)
            _write_json(attempt_dir / "completed.json", {
                "result": str(result_path),
                "result_sha256": _sha256_file(result_path),
                "completed_unix": time.time(),
            })
        except BaseException as error:
            _write_json(attempt_dir / "failure.json", {
                "stage": args.stage,
                "attempt_id": attempt_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": float(time.perf_counter() - started),
                "failed_unix": time.time(),
            })
            raise

    compact = {key: value for key, value in result.items()
               if key not in {"choices", "correct", "folds", "source_hashes",
                              "subgroups", "observed_reads", "integrity"}}
    print(json.dumps(compact, indent=2, ensure_ascii=False, sort_keys=True),
          flush=True)
    return result


if __name__ == "__main__":
    main()
