"""Nested train-only evaluation for the registered G3 group reranker."""

from pathlib import Path
import json
import time

import numpy as np

from crossfit import fusion_fit_stop_split, make_stratified_folds
from features import answer_metadata, extract_features
from group_rank_features import (build_group_rank_features,
                                 build_unique_group_labels, fit_pca)
from group_tree_ranker import (candidate_group_scores, fit_ranker, mixed_group_choices,
                               predict_ranker, scores_to_group_probabilities)
from models import fit_scorer, score_model


B0_CONFIG = {
    "architecture": "linear", "objective": "listwise",
    "learning_rate": 0.003, "l2": 0.0001, "batch_size": 128,
    "epochs": 12, "seed": 2028, "monitor_training_loss": False,
}
TREE_STRUCTURES = (
    {"depth": 4, "l2_leaf_reg": 10},
    {"depth": 4, "l2_leaf_reg": 30},
    {"depth": 6, "l2_leaf_reg": 10},
    {"depth": 6, "l2_leaf_reg": 30},
)
TREE_PREFIXES = (100, 200, 400, 600)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def registered_tree_config(depth, l2_leaf_reg, iterations, seed_offset=0):
    return {
        "depth": int(depth), "l2_leaf_reg": float(l2_leaf_reg),
        "iterations": int(iterations), "seed": 2028 + int(seed_offset),
    }


def _mixed_probabilities(b0_scores, tree_scores, valid, alpha):
    b0 = scores_to_group_probabilities(b0_scores, valid)
    tree = scores_to_group_probabilities(tree_scores, valid)
    return (1.0 - float(alpha)) * b0 + float(alpha) * tree


def _nll(probabilities, group_labels, hard):
    rows, slots = np.where(group_labels & hard[:, None])
    selected = np.maximum(probabilities[rows, slots], 1e-12)
    return float(-np.log(selected).mean()) if len(selected) else 0.0


def labels_from_representatives(candidate_labels, group_valid, representatives):
    candidate_labels = np.asarray(candidate_labels, dtype=bool)
    group_valid = np.asarray(group_valid, dtype=bool)
    representatives = np.asarray(representatives, dtype=np.int32)
    if group_valid.shape != representatives.shape or candidate_labels.shape != group_valid.shape:
        raise ValueError("candidate labels and group arrays are not aligned")
    result = np.zeros(group_valid.shape, dtype=bool)
    for row in range(len(result)):
        for slot in np.flatnonzero(group_valid[row]):
            representative = int(representatives[row, slot])
            if representative < 0:
                raise ValueError("valid group lacks a representative")
            result[row, slot] = candidate_labels[row, representative]
        if group_valid[row].any() and result[row, group_valid[row]].sum() != 1:
            raise ValueError("each supervised question must have exactly one correct group")
    return result


def select_inner_configuration(tree_predictions, b0_group_scores, group_valid,
                               representatives, candidate_labels, hard_mask,
                               alphas=ALPHAS):
    group_labels = labels_from_representatives(
        candidate_labels, group_valid, representatives)
    best = None
    for prediction in tree_predictions:
        for alpha in alphas:
            choices = mixed_group_choices(
                b0_group_scores, prediction["scores"], group_valid,
                representatives, alpha,
            )
            correct = candidate_labels[np.arange(len(candidate_labels)), choices]
            probabilities = _mixed_probabilities(
                b0_group_scores, prediction["scores"], group_valid, alpha)
            row = {
                "depth": int(prediction["depth"]),
                "l2_leaf_reg": float(prediction["l2_leaf_reg"]),
                "tree_count": int(prediction["tree_count"]),
                "alpha": float(alpha),
                "hard_correct": int(correct[hard_mask].sum()),
                "hard_n": int(hard_mask.sum()),
                "nll": _nll(probabilities, group_labels, hard_mask),
            }
            key = (-row["hard_correct"], row["nll"], row["alpha"],
                   row["tree_count"], row["depth"], -row["l2_leaf_reg"])
            if best is None or key < best[0]:
                best = (key, row)
    if best is None:
        raise ValueError("at least one tree prediction is required")
    return best[1]


def _metadata_slice(metadata, indices):
    return {key: np.asarray(value)[indices] for key, value in metadata.items()}


def _strata(metadata):
    result = []
    for row in range(len(metadata["valid"])):
        valid = metadata["valid"][row]
        groups = metadata["group_index"][row]
        group_count = len(set(groups[valid].tolist()))
        counts = [int(np.sum(groups[valid] == slot))
                  for slot in set(groups[valid].tolist())]
        result.append((group_count, max(counts) if counts else 0))
    return result


def _fit_b0(x, names, labels, valid, fit_indices, score_indices, seed_offset=0):
    config = {**B0_CONFIG, "seed": 2028 + int(seed_offset),
              "feature_names": list(names)}
    model = fit_scorer(x[fit_indices], labels[fit_indices], valid[fit_indices], config)
    return model, score_model(x[score_indices], model)


def _crossfit_b0(x, names, labels, valid, subset, strata, fold_seed,
                 seed_offset=0):
    subset = np.asarray(subset, dtype=np.int64)
    folds = make_stratified_folds(
        [str(index) for index in subset], [strata[index] for index in subset],
        n_splits=3, seed=int(fold_seed),
    )
    scores = np.zeros((len(subset), 4), dtype=np.float32)
    coverage = np.zeros(len(subset), dtype=np.int8)
    for local_held in folds:
        held = subset[local_held]
        local_fit = np.asarray(sorted(set(range(len(subset))) -
                                      set(local_held.tolist())), dtype=np.int64)
        fit = subset[local_fit]
        _, fold_scores = _fit_b0(
            x, names, labels, valid, fit, held, seed_offset=seed_offset)
        scores[local_held] = fold_scores
        coverage[local_held] += 1
    if np.any(coverage != 1):
        raise RuntimeError("B0 crossfit coverage must be exactly one")
    return scores


def _build_partition(bank, embeddings, candidate, candidate_names, metadata,
                     indices, b0_scores, pca):
    indices = np.asarray(indices, dtype=np.int64)
    return build_group_rank_features(
        [bank[index] for index in indices], embeddings[indices],
        candidate[indices], candidate_names, _metadata_slice(metadata, indices),
        b0_scores, pca,
    )


def _tree_search(train_features, train_labels, select_features, select_labels,
                 select_b0, select_hard, seed_offset=0):
    train_group_labels = labels_from_representatives(
        train_labels, train_features["group_valid"],
        train_features["representatives"])
    hard_train = train_features["group_valid"].sum(axis=1) >= 2
    predictions = []
    models = {}
    for structure in TREE_STRUCTURES:
        config = registered_tree_config(
            structure["depth"], structure["l2_leaf_reg"], 600,
            seed_offset=seed_offset)
        model = fit_ranker(
            train_features["values"][hard_train],
            train_features["group_valid"][hard_train],
            train_group_labels[hard_train], config,
        )
        models[(structure["depth"], structure["l2_leaf_reg"])] = model
        for prefix in TREE_PREFIXES:
            predictions.append({
                **structure, "tree_count": prefix,
                "scores": predict_ranker(
                    model, select_features["values"],
                    select_features["group_valid"], ntree_end=prefix),
            })
    selected = select_inner_configuration(
        predictions, select_b0, select_features["group_valid"],
        select_features["representatives"], select_labels, select_hard,
    )
    return selected


def _correct(labels, choices):
    return labels[np.arange(len(labels)), np.asarray(choices, dtype=np.int64)]


def run_group_tree_oof(bank, embeddings, labels, hard_mask, fold_seed=2026,
                       output_dir=None, observer=None):
    """Run the fixed nested five-fold G3 experiment and return aggregate evidence."""
    started = time.perf_counter()
    if int(fold_seed) not in {2026, 2027}:
        raise ValueError("registered OOF fold_seed must be 2026 or 2027")
    seed_offset = 0 if int(fold_seed) == 2026 else 10000
    labels = np.asarray(labels, dtype=bool)
    hard_mask = np.asarray(hard_mask, dtype=bool)
    metadata = answer_metadata(bank)
    candidate, candidate_names = extract_features(bank, embeddings, "embedding_text")
    keep = [i for i, name in enumerate(candidate_names)
            if name not in {"valid", "votes"}]
    b0_x = candidate[:, :, keep]
    b0_names = [candidate_names[i] for i in keep]
    strata = _strata(metadata)
    ids = [str(row["id"]) for row in bank]
    outer_folds = make_stratified_folds(ids, strata, n_splits=5, seed=fold_seed)
    b0_choices = np.zeros(len(bank), dtype=np.int64)
    g3_choices = np.zeros(len(bank), dtype=np.int64)
    coverage = np.zeros(len(bank), dtype=np.int8)
    fold_records = []
    all_indices = set(range(len(bank)))
    for outer_fold, held in enumerate(outer_folds):
        outer_started = time.perf_counter()
        outer_train = np.asarray(sorted(all_indices - set(held.tolist())), dtype=np.int64)
        inner_fit, inner_select = fusion_fit_stop_split(
            outer_train, strata, seed=3100 + seed_offset + outer_fold)
        inner_oof = _crossfit_b0(
            b0_x, b0_names, labels, metadata["valid"], inner_fit, strata,
            4100 + seed_offset + outer_fold, seed_offset=seed_offset)
        _, inner_select_scores = _fit_b0(
            b0_x, b0_names, labels, metadata["valid"], inner_fit, inner_select,
            seed_offset=seed_offset)
        inner_pca = fit_pca(embeddings[inner_fit])
        inner_train_features = _build_partition(
            bank, embeddings, candidate, candidate_names, metadata,
            inner_fit, inner_oof, inner_pca)
        inner_select_features = _build_partition(
            bank, embeddings, candidate, candidate_names, metadata,
            inner_select, inner_select_scores, inner_pca)
        select_groups = candidate_group_scores(
            inner_select_scores, metadata["valid"][inner_select],
            metadata["group_index"][inner_select])
        selected = _tree_search(
            inner_train_features, labels[inner_fit], inner_select_features,
            labels[inner_select], select_groups["scores"],
            hard_mask[inner_select], seed_offset=seed_offset)

        outer_oof = _crossfit_b0(
            b0_x, b0_names, labels, metadata["valid"], outer_train, strata,
            5100 + seed_offset + outer_fold, seed_offset=seed_offset)
        _, held_b0_scores = _fit_b0(
            b0_x, b0_names, labels, metadata["valid"], outer_train, held,
            seed_offset=seed_offset)
        outer_pca = fit_pca(embeddings[outer_train])
        outer_train_features = _build_partition(
            bank, embeddings, candidate, candidate_names, metadata,
            outer_train, outer_oof, outer_pca)
        held_features = _build_partition(
            bank, embeddings, candidate, candidate_names, metadata,
            held, held_b0_scores, outer_pca)
        outer_group_labels = labels_from_representatives(
            labels[outer_train], outer_train_features["group_valid"],
            outer_train_features["representatives"])
        outer_hard = outer_train_features["group_valid"].sum(axis=1) >= 2
        final_model = fit_ranker(
            outer_train_features["values"][outer_hard],
            outer_train_features["group_valid"][outer_hard],
            outer_group_labels[outer_hard],
            registered_tree_config(
                selected["depth"], selected["l2_leaf_reg"],
                selected["tree_count"], seed_offset=seed_offset),
        )
        held_tree_scores = predict_ranker(
            final_model, held_features["values"], held_features["group_valid"])
        held_groups = candidate_group_scores(
            held_b0_scores, metadata["valid"][held],
            metadata["group_index"][held])
        fold_b0 = mixed_group_choices(
            held_groups["scores"], held_tree_scores,
            held_groups["group_valid"], held_groups["representatives"], 0.0)
        fold_g3 = mixed_group_choices(
            held_groups["scores"], held_tree_scores,
            held_groups["group_valid"], held_groups["representatives"],
            selected["alpha"])
        b0_choices[held] = fold_b0
        g3_choices[held] = fold_g3
        coverage[held] += 1
        base_correct = _correct(labels[held], fold_b0)
        new_correct = _correct(labels[held], fold_g3)
        record = {
            "outer_fold": outer_fold,
            "tree_seed": 2028 + seed_offset,
            "held_count": int(len(held)),
            "hard_count": int(hard_mask[held].sum()),
            "selected": selected,
            "b0_correct": int(base_correct.sum()),
            "g3_correct": int(new_correct.sum()),
            "b0_hard_correct": int(base_correct[hard_mask[held]].sum()),
            "g3_hard_correct": int(new_correct[hard_mask[held]].sum()),
            "improvements": int((new_correct & ~base_correct).sum()),
            "regressions": int((~new_correct & base_correct).sum()),
            "wall_seconds": float(time.perf_counter() - outer_started),
        }
        fold_records.append(record)
        if observer:
            observer(record)
    if np.any(coverage != 1):
        raise RuntimeError("outer coverage must be exactly one")
    b0_correct = _correct(labels, b0_choices)
    g3_correct = _correct(labels, g3_choices)
    positive_folds = sum(
        row["g3_hard_correct"] > row["b0_hard_correct"] for row in fold_records)
    nonzero_alpha = sum(row["selected"]["alpha"] > 0 for row in fold_records)
    hard_delta = int(g3_correct[hard_mask].sum() - b0_correct[hard_mask].sum())
    summary = {
        "schema": 1, "evidence_role": "train_oof_development",
        "fold_seed": int(fold_seed), "validation_accessed": False,
        "seed_offset": int(seed_offset),
        "test_accessed": False, "coverage_exactly_one": True,
        "b0_overall_correct": int(b0_correct.sum()),
        "g3_overall_correct": int(g3_correct.sum()),
        "b0_hard_correct": int(b0_correct[hard_mask].sum()),
        "g3_hard_correct": int(g3_correct[hard_mask].sum()),
        "hard_n": int(hard_mask.sum()),
        "overall_n": int(len(bank)),
        "hard_delta": hard_delta,
        "improvements": int((g3_correct & ~b0_correct).sum()),
        "regressions": int((~g3_correct & b0_correct).sum()),
        "positive_folds": int(positive_folds),
        "nonzero_alpha_folds": int(nonzero_alpha),
        "primary_gate_passed": bool(
            hard_delta >= 10 and positive_folds >= 3 and nonzero_alpha >= 3),
        "wall_seconds": float(time.perf_counter() - started),
        "folds": fold_records,
        "b0_choices": b0_choices.astype(int).tolist(),
        "g3_choices": g3_choices.astype(int).tolist(),
    }
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"oof-seed-{fold_seed}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
    return summary
