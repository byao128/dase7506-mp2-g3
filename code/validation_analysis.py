"""Post-hoc, validation-only report analysis for the frozen original G3.

This script never fits a model and never accepts a test split.  It emits only
aggregate metrics: no per-question choices, probabilities, labels, or errors.
"""

import os


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _name in THREAD_ENVIRONMENT:
    os.environ[_name] = "4"

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


FORMAT = "g3-validation-report-analysis-v1"
METHOD_ORDER = (
    "full_alpha_0_5",
    "b0_group_alpha_0",
    "tree_group_alpha_1",
    "b0_candidate_argmax",
    "input_majority",
)
METHOD_LABELS = {
    "full_alpha_0_5": "Frozen G3: 0.5 B0 group probability + 0.5 tree group probability",
    "b0_group_alpha_0": "B0 after parse-equivalent answer-group aggregation",
    "tree_group_alpha_1": "Tree branch only at final fusion (tree inputs still include B0-derived features)",
    "b0_candidate_argmax": "B0 candidate-level argmax without group aggregation",
    "input_majority": "Input-only majority parsed-answer group",
}


def _correct_vector(candidate_labels, choices):
    labels = np.asarray(candidate_labels, dtype=bool)
    choices = np.asarray(choices, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != 4 or choices.shape != (len(labels),):
        raise ValueError("labels must be [N,4] and choices must be [N]")
    if np.any((choices < 0) | (choices > 3)):
        raise ValueError("choices must be candidate indices 0..3")
    return labels[np.arange(len(labels)), choices]


def accuracy_metrics(candidate_labels, choices, mask=None):
    correct = _correct_vector(candidate_labels, choices)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != correct.shape:
            raise ValueError("mask must align with choices")
        correct = correct[mask]
    n = int(len(correct))
    count = int(correct.sum())
    return {
        "correct": count,
        "n": n,
        "accuracy_percent": 100.0 * count / n if n else None,
    }


def paired_changes(candidate_labels, method_choices, comparator_choices, mask=None):
    method_choices = np.asarray(method_choices, dtype=np.int64)
    comparator_choices = np.asarray(comparator_choices, dtype=np.int64)
    method = _correct_vector(candidate_labels, method_choices)
    comparator = _correct_vector(candidate_labels, comparator_choices)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != method.shape:
            raise ValueError("mask must align with choices")
        method = method[mask]
        comparator = comparator[mask]
        method_choices = method_choices[mask]
        comparator_choices = comparator_choices[mask]
    improvements = int((method & ~comparator).sum())
    regressions = int((~method & comparator).sum())
    return {
        "n": int(len(method)),
        "improvements": improvements,
        "regressions": regressions,
        "net": improvements - regressions,
        "agreements": int((method == comparator).sum()),
        "choice_disagreements": int((method_choices != comparator_choices).sum()),
    }


def candidate_argmax_choices(scores, valid):
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if scores.shape != valid.shape or scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("candidate scores and validity must share shape [N,4]")
    result = np.zeros(len(scores), dtype=np.int64)
    for row in range(len(scores)):
        candidates = np.flatnonzero(valid[row])
        if len(candidates):
            result[row] = max(
                candidates, key=lambda index: (float(scores[row, index]), -int(index)))
    return result


def input_majority_choices(valid, groups):
    valid = np.asarray(valid, dtype=bool)
    groups = np.asarray(groups, dtype=np.int32)
    if valid.shape != groups.shape or valid.ndim != 2 or valid.shape[1] != 4:
        raise ValueError("validity and group indices must share shape [N,4]")
    result = np.zeros(len(valid), dtype=np.int64)
    for row in range(len(valid)):
        slots = sorted(set(groups[row, valid[row]].tolist()))
        if not slots:
            continue
        counts = {
            slot: int(np.sum(valid[row] & (groups[row] == slot))) for slot in slots
        }
        representatives = {
            slot: int(np.flatnonzero(valid[row] & (groups[row] == slot))[0])
            for slot in slots
        }
        chosen = max(slots, key=lambda slot: (counts[slot], -representatives[slot]))
        result[row] = representatives[chosen]
    return result


def fixed_bin_calibration(confidence, correct, bin_count=10):
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    if confidence.shape != correct.shape or confidence.ndim != 1:
        raise ValueError("confidence and correctness must be aligned vectors")
    if bin_count != 10:
        raise ValueError("this report protocol fixes ten equal-width bins")
    if np.any(~np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("confidence must be finite and within [0,1]")
    indices = np.minimum((confidence * bin_count).astype(np.int64), bin_count - 1)
    bins = []
    ece = 0.0
    n = len(confidence)
    for index in range(bin_count):
        selected = indices == index
        count = int(selected.sum())
        mean_confidence = float(confidence[selected].mean()) if count else None
        accuracy = float(correct[selected].mean()) if count else None
        if count:
            ece += count / n * abs(mean_confidence - accuracy)
        bins.append({
            "bin_index": index,
            "lower_inclusive": index / bin_count,
            "upper_inclusive": (index + 1) / bin_count if index == bin_count - 1 else None,
            "upper_exclusive": (index + 1) / bin_count if index < bin_count - 1 else None,
            "count": count,
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
        })
    return {
        "n": int(n),
        "bin_count": bin_count,
        "ece": float(ece) if n else None,
        "brier": float(np.mean((confidence - correct.astype(np.float64)) ** 2)) if n else None,
        "bins": bins,
    }


def _candidate_probabilities(scores, valid):
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    result = np.zeros_like(scores, dtype=np.float64)
    for row in range(len(scores)):
        indices = np.flatnonzero(valid[row])
        if not len(indices):
            result[row, 0] = 1.0
            continue
        shifted = scores[row, indices] - np.max(scores[row, indices])
        exponentiated = np.exp(shifted)
        result[row, indices] = exponentiated / exponentiated.sum()
    return result


def _chosen_candidate_confidence(probabilities, choices):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.int64)
    return probabilities[np.arange(len(choices)), choices]


def _chosen_group_confidence(probabilities, representatives, choices):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    representatives = np.asarray(representatives, dtype=np.int32)
    choices = np.asarray(choices, dtype=np.int64)
    confidence = np.zeros(len(choices), dtype=np.float64)
    for row, choice in enumerate(choices):
        slots = np.flatnonzero(representatives[row] == choice)
        if len(slots):
            confidence[row] = probabilities[row, slots[0]]
        else:
            confidence[row] = 1.0
    return confidence


def _majority_confidence(valid, groups, choices):
    valid = np.asarray(valid, dtype=bool)
    groups = np.asarray(groups, dtype=np.int32)
    choices = np.asarray(choices, dtype=np.int64)
    confidence = np.ones(len(choices), dtype=np.float64)
    for row, choice in enumerate(choices):
        denominator = int(valid[row].sum())
        if not denominator or not valid[row, choice]:
            continue
        slot = groups[row, choice]
        confidence[row] = float(np.sum(valid[row] & (groups[row] == slot))) / denominator
    return confidence


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_labels(bank, data_dir, read_jsonl, parse_answer):
    references = read_jsonl(Path(data_dir) / "gsm8k_validation.jsonl")
    if len(references) != len(bank):
        raise ValueError("validation references and candidates differ in length")
    labels = np.zeros((len(bank), 4), dtype=bool)
    for row_index, (row, reference) in enumerate(zip(bank, references)):
        if (reference.get("id") != row.get("id") or
                reference.get("question") != row.get("question")):
            raise ValueError("validation reference alignment mismatch")
        gold = parse_answer(reference.get("answer", ""))
        if gold is None:
            raise ValueError("unparseable validation reference")
        for candidate_index, candidate in enumerate(row["candidates"]):
            answer = parse_answer(candidate["text"])
            labels[row_index, candidate_index] = answer is not None and answer == gold
    return labels


def run_analysis(source_dir, data_dir, checkpoint, freeze_path):
    source_dir = Path(source_dir).resolve()
    data_dir = Path(data_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    freeze_path = Path(freeze_path).resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

    from common import check_data, load_inputs, parse_answer, read_jsonl
    from evaluate_group_tree_bundle import verify_freeze
    from features import answer_metadata, extract_features
    from group_rank_features import build_group_rank_features
    from group_tree_ranker import (
        candidate_group_scores,
        load_ranker,
        mixed_group_choices,
        predict_ranker,
        scores_to_group_probabilities,
    )
    from models import load_model, score_model
    from submission_group_tree import _load_pca

    freeze = verify_freeze(checkpoint, freeze_path)
    check_data(data_dir, ("validation",), inputs_only=False)
    bank, arrays = load_inputs("validation", data_dir)
    labels = _build_labels(bank, data_dir, read_jsonl, parse_answer)
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
    metadata = answer_metadata(bank)

    inference_path = checkpoint / "inference.json"
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    alpha = float(inference["config"]["alpha"])
    if alpha != 0.5:
        raise ValueError("the original frozen G3 report protocol requires alpha=0.5")
    candidate, names = extract_features(bank, embeddings, inference["feature_mode"])
    excluded = set(inference["exclude_features"])
    keep = [index for index, name in enumerate(names) if name not in excluded]
    selected_names = [names[index] for index in keep]
    if selected_names != inference["b0_feature_names"]:
        raise ValueError("B0 feature schema differs from the frozen checkpoint")
    b0_model = load_model(checkpoint / "b0")
    if selected_names != b0_model.get("feature_names"):
        raise ValueError("B0 model columns differ from inference.json")
    b0_scores = score_model(candidate[:, :, keep], b0_model)
    group_features = build_group_rank_features(
        bank, embeddings, candidate, names, metadata, b0_scores,
        _load_pca(checkpoint / "pca.npz"),
    )
    if group_features["values"].shape[2] != int(inference["group_feature_count"]):
        raise ValueError("G3 group feature schema differs from the frozen checkpoint")
    tree_scores = predict_ranker(
        load_ranker(checkpoint / "tree.json"),
        group_features["values"], group_features["group_valid"],
    )
    groups = candidate_group_scores(
        b0_scores, metadata["valid"], metadata["group_index"])
    group_valid = groups["group_valid"]
    representatives = groups["representatives"]

    choices = {
        "full_alpha_0_5": mixed_group_choices(
            groups["scores"], tree_scores, group_valid, representatives, 0.5),
        "b0_group_alpha_0": mixed_group_choices(
            groups["scores"], tree_scores, group_valid, representatives, 0.0),
        "tree_group_alpha_1": mixed_group_choices(
            groups["scores"], tree_scores, group_valid, representatives, 1.0),
        "b0_candidate_argmax": candidate_argmax_choices(
            b0_scores, metadata["valid"]),
        "input_majority": input_majority_choices(
            metadata["valid"], metadata["group_index"]),
    }

    b0_group_probabilities = scores_to_group_probabilities(groups["scores"], group_valid)
    tree_probabilities = scores_to_group_probabilities(tree_scores, group_valid)
    full_probabilities = 0.5 * b0_group_probabilities + 0.5 * tree_probabilities
    candidate_probabilities = _candidate_probabilities(b0_scores, metadata["valid"])
    confidence = {
        "full_alpha_0_5": _chosen_group_confidence(
            full_probabilities, representatives, choices["full_alpha_0_5"]),
        "b0_group_alpha_0": _chosen_group_confidence(
            b0_group_probabilities, representatives, choices["b0_group_alpha_0"]),
        "tree_group_alpha_1": _chosen_group_confidence(
            tree_probabilities, representatives, choices["tree_group_alpha_1"]),
        "b0_candidate_argmax": _chosen_candidate_confidence(
            candidate_probabilities, choices["b0_candidate_argmax"]),
        "input_majority": _majority_confidence(
            metadata["valid"], metadata["group_index"], choices["input_majority"]),
    }

    hard = np.asarray(metadata["hard_mask"], dtype=bool)
    methods = {}
    for method in METHOD_ORDER:
        hard_correct = _correct_vector(labels, choices[method])[hard]
        methods[method] = {
            "label": METHOD_LABELS[method],
            "all": accuracy_metrics(labels, choices[method]),
            "hard": accuracy_metrics(labels, choices[method], hard),
            "hard_calibration_fixed_10_bins": fixed_bin_calibration(
                confidence[method][hard], hard_correct),
        }

    paired = []
    for left_index, left in enumerate(METHOD_ORDER):
        for right in METHOD_ORDER[left_index + 1:]:
            paired.append({
                "method": left,
                "comparator": right,
                "all": paired_changes(labels, choices[left], choices[right]),
                "hard": paired_changes(labels, choices[left], choices[right], hard),
            })

    return {
        "format": FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_role": "retrospective_posthoc_validation_report_analysis",
        "not_preregistered": True,
        "selection_or_tuning_use": "prohibited; frozen G3 and submitted score must not change from this analysis",
        "split": "validation",
        "test_accessed": False,
        "model_fit_performed": False,
        "per_example_output_written": False,
        "fixed_components": list(METHOD_ORDER),
        "counts": {
            "all": int(len(bank)),
            "hard": int(hard.sum()),
            "candidate_count_per_question": 4,
        },
        "methods": methods,
        "paired_changes": paired,
        "provenance": {
            "checkpoint_format": freeze["format"],
            "checkpoint_asset_bytes": int(freeze["asset_bytes"]),
            "freeze_sha256": _sha256(freeze_path),
            "inference_json_sha256": _sha256(inference_path),
            "validation_manifest_verified_by_original_common_check_data": True,
            "label_parser": "original common.parse_answer",
            "group_schema": "original features.answer_metadata",
            "tie_break": "earliest candidate index, matching original group selection",
            "thread_environment": {name: os.environ[name] for name in THREAD_ENVIRONMENT},
        },
        "interpretation_limits": [
            "These are retrospective validation analyses for explanation, not pre-registered tests.",
            "Component differences are paired associations on one validation split, not causal proof.",
            "Tree alpha=1 removes only the final B0 probability mixture; the tree still consumes B0-derived group features.",
            "B0 candidate argmax versus B0 group alpha=0 isolates output grouping/aggregation under the same B0 candidate scores.",
            "Calibration is descriptive on only 355 hard validation questions with ten fixed equal-width bins; no calibrator was fit.",
            "The reported score-derived confidence is not claimed to be a calibrated posterior probability.",
            "The analysis cannot justify changing the frozen method, checkpoint, alpha, or test submission choice.",
        ],
    }


def build_plot_data(result):
    summary_rows = []
    calibration_rows = []
    for method in METHOD_ORDER:
        record = result["methods"][method]
        for slice_name in ("all", "hard"):
            summary_rows.append({
                "method": method,
                "method_label": record["label"],
                "slice": slice_name,
                **record[slice_name],
            })
        for row in record["hard_calibration_fixed_10_bins"]["bins"]:
            calibration_rows.append({"method": method, "slice": "hard", **row})
    paired_rows = []
    for comparison in result["paired_changes"]:
        for slice_name in ("all", "hard"):
            paired_rows.append({
                "method": comparison["method"],
                "comparator": comparison["comparator"],
                "slice": slice_name,
                **comparison[slice_name],
            })
    return {
        "format": "g3-validation-report-plot-data-v1",
        "evidence_role": result["evidence_role"],
        "contains_per_example_data": False,
        "summary_rows": summary_rows,
        "paired_rows": paired_rows,
        "calibration_rows": calibration_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-data", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.plot_data.resolve():
        raise ValueError("--output and --plot-data must differ")
    result = run_analysis(
        args.source_dir, args.data_dir, args.checkpoint, args.freeze)
    plot_data = build_plot_data(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.plot_data.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.plot_data.write_text(
        json.dumps(plot_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "format": result["format"],
        "split": result["split"],
        "counts": result["counts"],
        "methods": {
            name: {"all": result["methods"][name]["all"],
                   "hard": result["methods"][name]["hard"]}
            for name in METHOD_ORDER
        },
        "output": str(args.output),
        "plot_data": str(args.plot_data),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
