"""Frozen final-fit and one-shot validation evaluation for G3."""

from collections import Counter
from pathlib import Path
import json
import time

import numpy as np

from features import answer_metadata, extract_features
from group_rank_features import fit_pca
from group_tree_cv import (
    _build_partition, _crossfit_b0, _strata,
    candidate_group_scores, labels_from_representatives,
    registered_tree_config,
)
from group_tree_ranker import fit_ranker, mixed_group_choices, predict_ranker


def validate_final_gates(primary, stability):
    """Reject validation access unless both registered train-only gates passed."""
    required_primary = {
        "evidence_role": "train_oof_development",
        "fold_seed": 2026,
        "primary_gate_passed": True,
        "coverage_exactly_one": True,
        "validation_accessed": False,
        "test_accessed": False,
    }
    required_stability = {
        "evidence_role": "train_oof_development",
        "fold_seed": 2027,
        "coverage_exactly_one": True,
        "validation_accessed": False,
        "test_accessed": False,
    }
    for name, payload, required in (
            ("primary", primary, required_primary),
            ("stability", stability, required_stability)):
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"{name} gate requires {key}={expected!r}")
    if int(stability.get("hard_delta", 0)) <= 0:
        raise ValueError("stability gate requires hard_delta > 0")


def modal_configuration(folds):
    tuples = [
        (int(row["selected"]["depth"]),
         float(row["selected"]["l2_leaf_reg"]),
         int(row["selected"]["tree_count"]),
         float(row["selected"]["alpha"]))
        for row in folds
    ]
    if not tuples:
        raise ValueError("primary folds are required")
    counts = Counter(tuples)
    maximum = max(counts.values())
    tied = [item for item, count in counts.items() if count == maximum]
    depth, l2, trees, alpha = min(
        tied, key=lambda item: (item[3], item[2], item[0], -item[1]))
    return {
        "depth": depth, "l2_leaf_reg": l2,
        "tree_count": trees, "alpha": alpha,
    }


def _correct(labels, choices):
    return labels[np.arange(len(labels)), np.asarray(choices, dtype=np.int64)]


def fit_final_and_score_validation(train_bank, train_embeddings, train_labels,
                                   validation_bank, validation_embeddings,
                                   validation_labels, config, output_file=None):
    """Fit on all train and score one frozen validation candidate."""
    started = time.perf_counter()
    train_metadata = answer_metadata(train_bank)
    validation_metadata = answer_metadata(validation_bank)
    train_candidate, candidate_names = extract_features(
        train_bank, train_embeddings, "embedding_text")
    validation_candidate, validation_names = extract_features(
        validation_bank, validation_embeddings, "embedding_text")
    if validation_names != candidate_names:
        raise RuntimeError("train and validation feature schemas differ")
    keep = [i for i, name in enumerate(candidate_names)
            if name not in {"valid", "votes"}]
    b0_names = [candidate_names[i] for i in keep]
    train_b0_x = train_candidate[:, :, keep]
    validation_b0_x = validation_candidate[:, :, keep]
    all_train = np.arange(len(train_bank), dtype=np.int64)
    train_strata = _strata(train_metadata)
    train_b0_oof = _crossfit_b0(
        train_b0_x, b0_names, train_labels, train_metadata["valid"],
        all_train, train_strata, 6100, seed_offset=0)
    # The train and validation tensors are distinct, so fit once and score the
    # validation tensor explicitly rather than passing foreign indices.
    from models import fit_scorer, score_model
    from group_tree_cv import B0_CONFIG
    b0_model = fit_scorer(
        train_b0_x, train_labels, train_metadata["valid"],
        {**B0_CONFIG, "feature_names": b0_names})
    validation_b0_scores = score_model(validation_b0_x, b0_model)

    pca = fit_pca(train_embeddings)
    train_features = _build_partition(
        train_bank, train_embeddings, train_candidate, candidate_names,
        train_metadata, all_train, train_b0_oof, pca)
    validation_features = _build_partition(
        validation_bank, validation_embeddings, validation_candidate,
        validation_names, validation_metadata,
        np.arange(len(validation_bank), dtype=np.int64),
        validation_b0_scores, pca)
    train_group_labels = labels_from_representatives(
        train_labels, train_features["group_valid"],
        train_features["representatives"])
    hard_train = train_features["group_valid"].sum(axis=1) >= 2
    tree_model = fit_ranker(
        train_features["values"][hard_train],
        train_features["group_valid"][hard_train],
        train_group_labels[hard_train],
        registered_tree_config(
            config["depth"], config["l2_leaf_reg"],
            config["tree_count"], seed_offset=0),
    )
    validation_tree_scores = predict_ranker(
        tree_model, validation_features["values"],
        validation_features["group_valid"])
    validation_groups = candidate_group_scores(
        validation_b0_scores, validation_metadata["valid"],
        validation_metadata["group_index"])
    b0_choices = mixed_group_choices(
        validation_groups["scores"], validation_tree_scores,
        validation_groups["group_valid"],
        validation_groups["representatives"], 0.0)
    g3_choices = mixed_group_choices(
        validation_groups["scores"], validation_tree_scores,
        validation_groups["group_valid"],
        validation_groups["representatives"], config["alpha"])
    hard = validation_metadata["hard_mask"]
    b0_correct = _correct(validation_labels, b0_choices)
    g3_correct = _correct(validation_labels, g3_choices)
    result = {
        "schema": 1, "evidence_role": "validation_development",
        "config": dict(config), "validation_accessed": True,
        "test_accessed": False, "validation_n": int(len(validation_bank)),
        "validation_hard_n": int(hard.sum()),
        "b0_correct": int(b0_correct.sum()),
        "g3_correct": int(g3_correct.sum()),
        "b0_hard_correct": int(b0_correct[hard].sum()),
        "g3_hard_correct": int(g3_correct[hard].sum()),
        "improvements": int((g3_correct & ~b0_correct).sum()),
        "regressions": int((~g3_correct & b0_correct).sum()),
        "wall_seconds": float(time.perf_counter() - started),
        "b0_choices": b0_choices.astype(int).tolist(),
        "g3_choices": g3_choices.astype(int).tolist(),
    }
    if output_file:
        Path(output_file).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    return result
