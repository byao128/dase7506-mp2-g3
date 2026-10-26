"""Fixed paired train-only OOF experiment for G3-single and G3-BAG3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from crossfit import array_sha256, canonical_sha256, make_stratified_folds
from features import answer_metadata, extract_features
from group_rank_features import fit_pca
from group_tree_cv import (
    _build_partition,
    _crossfit_b0,
    _fit_b0,
    _strata,
    labels_from_representatives,
    registered_tree_config,
)
from group_tree_ranker import (
    candidate_group_scores,
    fit_ranker,
    mixed_group_choices,
    predict_ranker,
)


EXPERIMENT = "G3-query-bagging"
SCHEMA = 1
ALLOWED_FOLD_SEEDS = (6061, 6062)
FIXED_CONFIG = {
    "outer_splits": 5,
    "b0_inner_splits": 3,
    "pca_dimensions": 64,
    "g3_dimensions": 1438,
    "alpha": 0.5,
    "b0_seed_base": 2028,
    "b0_crossfit_seed_base": 8100,
    "single_tree_seed_base": 2028,
    "bag_tree_seed_bases": [2028, 2029, 2030],
    "bootstrap_seed_bases": [82601, 82602, 82603],
    "stage_seed_offsets": {"primary": 0, "stability": 10000},
    "fold_seeds": {"primary": 6061, "stability": 6062},
    "tree": {
        "loss_function": "QuerySoftMax",
        "iterations": 600,
        "depth": 4,
        "l2_leaf_reg": 10.0,
        "learning_rate": 0.03,
        "bootstrap_type": "No",
        "thread_count": 4,
    },
}


def _integer_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer question count")
    return int(value)


def evaluate_acceptance(stage, hard_delta, fold_hard_deltas,
                        candidate_overall_correct, g3_overall_correct,
                        b0_overall_correct):
    """Apply the pre-registered integer-count acceptance gate."""
    if stage not in {"primary", "stability"}:
        raise ValueError("stage must be primary or stability")
    hard_delta = _integer_count(hard_delta, "hard_delta")
    deltas = [_integer_count(value, "fold_hard_delta")
              for value in fold_hard_deltas]
    if len(deltas) != 5:
        raise ValueError("the fixed gate requires five fold deltas")
    candidate = _integer_count(candidate_overall_correct,
                               "candidate_overall_correct")
    single = _integer_count(g3_overall_correct, "g3_overall_correct")
    b0 = _integer_count(b0_overall_correct, "b0_overall_correct")
    threshold = 10 if stage == "primary" else 5
    conditions = {
        "hard_delta_minimum": hard_delta >= threshold,
        "at_least_three_positive_folds": sum(value > 0 for value in deltas) >= 3,
        "worst_fold_at_least_minus_three": min(deltas) >= -3,
        "overall_not_below_single": candidate >= single,
        "overall_not_below_b0": candidate >= b0,
    }
    passed = bool(all(conditions.values()))
    return {
        "stage": stage,
        "hard_delta_threshold": threshold,
        "hard_delta": hard_delta,
        "fold_hard_deltas": deltas,
        "positive_folds": int(sum(value > 0 for value in deltas)),
        "worst_fold_delta": int(min(deltas)),
        "candidate_overall_correct": candidate,
        "g3_overall_correct": single,
        "b0_overall_correct": b0,
        "conditions": conditions,
        "passed": passed,
        "gate_passed": passed,
    }


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    code = Path(__file__).resolve().parent
    root = code.parent
    paths = [
        *[code / name for name in (
            "common.py", "crossfit.py", "features.py", "models.py",
            "supervision.py", "verifier.py", "group_fusion.py",
            "group_rank_features.py",
            "group_tree_ranker.py", "group_tree_cv.py",
            "group_tree_bagging.py", "group_tree_bagging_cv.py",
            "run_group_tree_bagging.py",
        )],
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"source binding paths are missing: {missing}")
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in paths
    }


def _input_hashes(bank, embeddings, labels, hard_mask):
    return {
        "bank_content": canonical_sha256(bank),
        "bank_ids": canonical_sha256([str(row["id"]) for row in bank]),
        "embeddings": array_sha256(embeddings),
        "labels": array_sha256(labels),
        "hard_mask": array_sha256(hard_mask),
    }


def _validate_inputs(bank, embeddings, labels, fold_seed):
    if (isinstance(fold_seed, (bool, np.bool_)) or
            not isinstance(fold_seed, (int, np.integer)) or
            int(fold_seed) not in ALLOWED_FOLD_SEEDS):
        raise ValueError("fold_seed must be 6061 or 6062")
    n = len(bank)
    if n < FIXED_CONFIG["outer_splits"]:
        raise ValueError("at least five rows are required for fixed outer folds")
    ids = [str(row.get("id")) for row in bank]
    if len(ids) != len(set(ids)):
        raise ValueError("bank IDs must be unique")

    embedding_array = np.asarray(embeddings)
    if (embedding_array.shape != (n, 4, 1024) or
            not np.issubdtype(embedding_array.dtype, np.number) or
            np.issubdtype(embedding_array.dtype, np.complexfloating) or
            not np.isfinite(embedding_array).all()):
        raise ValueError("embeddings must be finite real numeric [N, 4, 1024]")
    label_array = np.asarray(labels)
    if label_array.dtype != np.bool_ or label_array.shape != (n, 4):
        raise ValueError("labels must be boolean [N, 4]")

    metadata = answer_metadata(bank)
    valid = np.asarray(metadata["valid"])
    groups = np.asarray(metadata["group_index"])
    if valid.dtype != np.bool_ or valid.shape != (n, 4) or groups.shape != (n, 4):
        raise ValueError("official answer metadata is not aligned")
    if np.any(label_array & ~valid):
        raise ValueError("positive labels must have valid parsed answers")
    strata = _strata(metadata)
    hard_mask = np.asarray([group_count >= 2 for group_count, _ in strata], dtype=bool)
    for row in range(n):
        correct_groups = set(groups[row, label_array[row] & valid[row]].tolist())
        if len(correct_groups) != 1:
            raise ValueError("labels must identify exactly one parsed answer group per row")
    declared_hard = metadata.get("hard_mask")
    if declared_hard is not None and not np.array_equal(
            hard_mask, np.asarray(declared_hard, dtype=bool)):
        raise ValueError("official hard mask disagrees with parsed answer groups")
    return (embedding_array.astype(np.float32, copy=False), label_array,
            metadata, strata, hard_mask, ids)


def _tree_config(seed):
    config = registered_tree_config(4, 10, 600, seed_offset=int(seed) - 2028)
    if config != {
            "depth": 4, "l2_leaf_reg": 10.0,
            "iterations": 600, "seed": int(seed)}:
        raise RuntimeError("registered tree configuration changed")
    return config


def _correct(labels, choices):
    choices = np.asarray(choices, dtype=np.int64)
    if choices.shape != (len(labels),) or np.any((choices < 0) | (choices >= 4)):
        raise ValueError("choices must identify one existing candidate per row")
    return np.asarray(labels, dtype=bool)[np.arange(len(labels)), choices]


def _metrics(labels, choices, hard_mask, subset=None):
    correct = _correct(labels, choices)
    selected = (np.ones(len(correct), dtype=bool) if subset is None
                else np.asarray(subset, dtype=bool))
    if selected.shape != (len(correct),):
        raise ValueError("metric subset is not aligned")
    hard_selected = selected & np.asarray(hard_mask, dtype=bool)
    return {
        "overall_correct": int(correct[selected].sum()),
        "overall_n": int(selected.sum()),
        "hard_correct": int(correct[hard_selected].sum()),
        "hard_n": int(hard_selected.sum()),
    }


def _paired(labels, candidate, control, hard_mask, subset=None):
    candidate_correct = _correct(labels, candidate)
    control_correct = _correct(labels, control)
    selected = (np.ones(len(candidate_correct), dtype=bool) if subset is None
                else np.asarray(subset, dtype=bool))
    hard_selected = selected & np.asarray(hard_mask, dtype=bool)
    return {
        "n": int(selected.sum()),
        "hard_n": int(hard_selected.sum()),
        "corrections": int((candidate_correct & ~control_correct & selected).sum()),
        "regressions": int((~candidate_correct & control_correct & selected).sum()),
        "hard_corrections": int(
            (candidate_correct & ~control_correct & hard_selected).sum()),
        "hard_regressions": int(
            (~candidate_correct & control_correct & hard_selected).sum()),
    }


def _disagreement(left, right, hard_mask, subset=None):
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    selected = (np.ones(len(left), dtype=bool) if subset is None
                else np.asarray(subset, dtype=bool))
    hard_selected = selected & np.asarray(hard_mask, dtype=bool)
    differs = left != right
    n = int(selected.sum())
    hard_n = int(hard_selected.sum())
    return {
        "count": int((differs & selected).sum()),
        "n": n,
        "rate": float((differs & selected).sum() / n) if n else 0.0,
        "hard_count": int((differs & hard_selected).sum()),
        "hard_n": hard_n,
        "hard_rate": float((differs & hard_selected).sum() / hard_n)
        if hard_n else 0.0,
    }


def _majority_classes(metadata, labels):
    valid = np.asarray(metadata["valid"], dtype=bool)
    groups = np.asarray(metadata["group_index"])
    result = []
    for row in range(len(valid)):
        slots = sorted(set(groups[row, valid[row]].tolist()))
        counts = [(slot, int(np.sum(valid[row] & (groups[row] == slot))))
                  for slot in slots]
        maximum = max((count for _, count in counts), default=0)
        leaders = [slot for slot, count in counts if count == maximum]
        if len(leaders) != 1:
            result.append("tie")
            continue
        correct_groups = set(groups[row, np.asarray(labels[row], dtype=bool) & valid[row]].tolist())
        result.append("unique_majority_correct" if leaders[0] in correct_groups
                      else "unique_majority_incorrect")
    return np.asarray(result, dtype=object)


def _subgroup_report(metadata, labels, hard_mask, choices):
    strata = _strata(metadata)
    group_counts = np.asarray([value[0] for value in strata], dtype=np.int64)
    majority = _majority_classes(metadata, labels)
    definitions = {
        **{f"group_count_{count}": group_counts == count for count in (2, 3, 4)},
        **{name: majority == name for name in (
            "unique_majority_correct", "unique_majority_incorrect", "tie")},
    }
    result = {}
    for name, subset in definitions.items():
        result[name] = {
            "metrics": {arm: _metrics(labels, value, hard_mask, subset)
                        for arm, value in choices.items()},
            "bag3_vs_single": _paired(
                labels, choices["bag3"], choices["g3_single"], hard_mask, subset),
            "bag3_vs_b0": _paired(
                labels, choices["bag3"], choices["b0"], hard_mask, subset),
        }
    return result


def validate_primary_for_stability(primary, input_hashes, source_hashes,
                                   config_sha256, n):
    """Recompute the fixed BAG3 primary decision before stability fitting."""
    if (primary.get("experiment") != EXPERIMENT or primary.get("schema") != SCHEMA or
            primary.get("stage") != "primary" or primary.get("fold_seed") != 6061):
        raise ValueError("primary result identity is invalid")
    if primary.get("input_hashes") != input_hashes:
        raise ValueError("primary input binding does not match stability")
    if primary.get("source_hashes") != source_hashes:
        raise ValueError("primary source binding does not match stability")
    if primary.get("config_sha256") != config_sha256:
        raise ValueError("primary fixed config binding does not match stability")
    if (primary.get("coverage_exactly_one") is not True or
            primary.get("coverage_counts") != [1] * int(n)):
        raise ValueError("primary outer coverage is invalid")
    metrics = primary.get("metrics", {})
    if not {"bag3", "g3_single", "b0"}.issubset(metrics):
        raise ValueError("primary result lacks fixed comparison metrics")
    folds = primary.get("folds", [])
    if len(folds) != 5:
        raise ValueError("primary result lacks five fold records")
    deltas = [row["bag3_hard_delta_vs_single"] for row in folds]
    recomputed = evaluate_acceptance(
        "primary",
        metrics["bag3"]["hard_correct"] -
        metrics["g3_single"]["hard_correct"],
        deltas,
        metrics["bag3"]["overall_correct"],
        metrics["g3_single"]["overall_correct"],
        metrics["b0"]["overall_correct"],
    )
    if (primary.get("gate") != recomputed or
            primary.get("gate_passed") is not recomputed["passed"] or
            recomputed["passed"] is not True):
        raise ValueError("primary gate failed or does not match recomputation")
    return True


def _emit(observer, event, **payload):
    if observer is not None:
        observer({"event": event, **payload})


def run_bagging_oof(bank, embeddings, labels, *, fold_seed,
                    output_dir=None, observer=None):
    """Run the fixed five-fold paired G3-single/BAG3 train OOF experiment."""
    from group_tree_bagging import (bagged_group_choices,
                                    stratified_query_bootstrap)

    started = time.perf_counter()
    embeddings, labels, metadata, strata, hard_mask, ids = _validate_inputs(
        bank, embeddings, labels, fold_seed)
    stage = "primary" if int(fold_seed) == 6061 else "stability"
    seed_offset = FIXED_CONFIG["stage_seed_offsets"][stage]
    source_hashes = _source_hashes()
    input_hashes = _input_hashes(bank, embeddings, labels, hard_mask)
    config_sha256 = canonical_sha256(FIXED_CONFIG)
    if stage == "stability":
        if output_dir is None:
            raise ValueError("stability requires output_dir containing frozen primary")
        primary_path = Path(output_dir) / "primary" / "primary.json"
        if not primary_path.is_file():
            raise FileNotFoundError("stability requires completed primary result")
        validate_primary_for_stability(
            json.loads(primary_path.read_text(encoding="utf-8")),
            input_hashes, source_hashes, config_sha256, len(bank))

    candidate, candidate_names = extract_features(bank, embeddings, "embedding_text")
    keep = [index for index, name in enumerate(candidate_names)
            if name not in {"valid", "votes"}]
    b0_x = candidate[:, :, keep]
    b0_names = [candidate_names[index] for index in keep]
    outer_folds = make_stratified_folds(
        ids, strata, n_splits=5, seed=int(fold_seed))
    all_rows = set(range(len(bank)))
    choices = {name: np.full(len(bank), -1, dtype=np.int64)
               for name in ("b0", "g3_single", "bag3", "member_0",
                            "member_1", "member_2")}
    coverage = np.zeros(len(bank), dtype=np.int8)
    folds = []
    copy_controls = []
    _emit(observer, "stage_start", stage=stage, fold_seed=int(fold_seed))

    for fold, held in enumerate(outer_folds):
        fold_started = time.perf_counter()
        held = np.asarray(held, dtype=np.int64)
        outer_train = np.asarray(sorted(all_rows - set(held.tolist())), dtype=np.int64)
        if len(held) == 0:
            raise RuntimeError("fixed outer folds must not be empty")
        _emit(observer, "fold_start", stage=stage, fold=fold,
              outer_train_n=int(len(outer_train)), held_n=int(len(held)))

        b0_crossfit_seed = 8100 + seed_offset + fold
        outer_oof = _crossfit_b0(
            b0_x, b0_names, labels, metadata["valid"], outer_train, strata,
            b0_crossfit_seed, seed_offset=seed_offset)
        _emit(observer, "b0_crossfit_complete", stage=stage, fold=fold,
              seed=b0_crossfit_seed)
        _, held_b0_scores = _fit_b0(
            b0_x, b0_names, labels, metadata["valid"], outer_train, held,
            seed_offset=seed_offset)
        _emit(observer, "b0_fit_complete", stage=stage, fold=fold,
              seed=2028 + seed_offset)
        pca = fit_pca(embeddings[outer_train])
        _emit(observer, "pca_fit_complete", stage=stage, fold=fold,
              fit_rows=int(len(outer_train)))
        train_features = _build_partition(
            bank, embeddings, candidate, candidate_names, metadata,
            outer_train, outer_oof, pca)
        held_features = _build_partition(
            bank, embeddings, candidate, candidate_names, metadata,
            held, held_b0_scores, pca)
        if (train_features["values"].shape[2] != 1438 or
                held_features["values"].shape[2] != 1438):
            raise RuntimeError("G3 feature schema must remain 1438 columns")
        train_hard = np.asarray(
            train_features["group_valid"].sum(axis=1) >= 2, dtype=bool)
        if not np.array_equal(train_hard, hard_mask[outer_train]):
            raise RuntimeError("partition hard mask changed after feature construction")
        hard_local = np.flatnonzero(train_hard).astype(np.int64)
        if not len(hard_local):
            raise RuntimeError("each outer training partition requires hard queries")
        group_labels = labels_from_representatives(
            labels[outer_train], train_features["group_valid"],
            train_features["representatives"])

        single_seed = 2028 + seed_offset
        single_model = fit_ranker(
            train_features["values"][hard_local],
            train_features["group_valid"][hard_local],
            group_labels[hard_local], _tree_config(single_seed))
        single_scores = predict_ranker(
            single_model, held_features["values"], held_features["group_valid"])
        _emit(observer, "single_complete", stage=stage, fold=fold,
              seed=single_seed)

        member_scores = []
        member_records = []
        local_strata = [strata[int(index)] for index in outer_train]
        for member, (tree_base, bootstrap_base) in enumerate(zip(
                FIXED_CONFIG["bag_tree_seed_bases"],
                FIXED_CONFIG["bootstrap_seed_bases"])):
            tree_seed = tree_base + seed_offset
            bootstrap_seed = bootstrap_base + seed_offset + 100 * fold
            draw_local = stratified_query_bootstrap(
                local_strata, hard_local, bootstrap_seed)
            draw_global = outer_train[draw_local]
            if (len(draw_local) != len(hard_local) or
                    not set(draw_global.tolist()).issubset(
                        set(outer_train[hard_local].tolist()))):
                raise RuntimeError("bootstrap escaped the outer-train hard subset")
            _emit(observer, "member_start", stage=stage, fold=fold,
                  member=member, tree_seed=tree_seed,
                  bootstrap_seed=bootstrap_seed)
            model = fit_ranker(
                train_features["values"][draw_local],
                train_features["group_valid"][draw_local],
                group_labels[draw_local], _tree_config(tree_seed))
            scores = predict_ranker(
                model, held_features["values"], held_features["group_valid"])
            member_scores.append(scores)
            member_records.append({
                "member": member,
                "tree_seed": tree_seed,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_local_rows": draw_local.astype(int).tolist(),
                "bootstrap_local_rows_sha256": array_sha256(draw_local),
                "bootstrap_global_rows": draw_global.astype(int).tolist(),
                "bootstrap_global_rows_sha256": array_sha256(draw_global),
                "fresh_query_ids": list(range(len(draw_local))),
                "fresh_query_ids_sha256": array_sha256(
                    np.arange(len(draw_local), dtype=np.int64)),
            })
            _emit(observer, "member_complete", stage=stage, fold=fold,
                  member=member, tree_seed=tree_seed,
                  bootstrap_seed=bootstrap_seed)
        stacked_scores = np.stack(member_scores, axis=0)
        held_groups = candidate_group_scores(
            held_b0_scores, metadata["valid"][held],
            metadata["group_index"][held])
        fold_b0 = mixed_group_choices(
            held_groups["scores"], single_scores,
            held_groups["group_valid"], held_groups["representatives"], 0.0)
        fold_single = mixed_group_choices(
            held_groups["scores"], single_scores,
            held_groups["group_valid"], held_groups["representatives"], 0.5)
        fold_bag = bagged_group_choices(
            held_groups["scores"], stacked_scores,
            held_groups["group_valid"], held_groups["representatives"], 0.5)
        fold_members = [bagged_group_choices(
            held_groups["scores"], stacked_scores[index:index + 1],
            held_groups["group_valid"], held_groups["representatives"], 0.5)
            for index in range(3)]
        copy_choice = bagged_group_choices(
            held_groups["scores"], np.stack([single_scores] * 3, axis=0),
            held_groups["group_valid"], held_groups["representatives"], 0.5)
        copy_equal = bool(np.array_equal(copy_choice, fold_single))
        if not copy_equal:
            raise RuntimeError("three identical tree copies must equal G3-single")
        copy_controls.append(copy_equal)

        fold_values = {
            "b0": fold_b0, "g3_single": fold_single, "bag3": fold_bag,
            **{f"member_{index}": value
               for index, value in enumerate(fold_members)},
        }
        nonhard = ~hard_mask[held]
        if any(not np.array_equal(value[nonhard], fold_single[nonhard])
               for value in fold_values.values()):
            raise RuntimeError("non-hard choices must match G3-single")
        for name, value in fold_values.items():
            choices[name][held] = value
        coverage[held] += 1

        fold_metrics = {
            name: _metrics(labels[held], value, hard_mask[held])
            for name, value in fold_values.items()
        }
        for member, member_record in enumerate(member_records):
            member_record["metrics"] = fold_metrics[f"member_{member}"]
        disagreements = {
            f"member_{left}_vs_member_{right}": _disagreement(
                fold_members[left], fold_members[right], hard_mask[held])
            for left, right in ((0, 1), (0, 2), (1, 2))
        }
        record = {
            "fold": fold,
            "outer_train_rows": outer_train.astype(int).tolist(),
            "outer_train_rows_sha256": array_sha256(outer_train),
            "held_rows": held.astype(int).tolist(),
            "held_rows_sha256": array_sha256(held),
            "b0_fit_rows": outer_train.astype(int).tolist(),
            "b0_fit_rows_sha256": array_sha256(outer_train),
            "pca_fit_rows": outer_train.astype(int).tolist(),
            "pca_fit_rows_sha256": array_sha256(outer_train),
            "b0_seed": 2028 + seed_offset,
            "b0_crossfit_seed": b0_crossfit_seed,
            "single_tree_seed": single_seed,
            "members": member_records,
            "hard_n": int(hard_mask[held].sum()),
            "overall_n": int(len(held)),
            "metrics": fold_metrics,
            "choices": {name: value.astype(int).tolist()
                        for name, value in fold_values.items()},
            "correct": {name: _correct(labels[held], value).astype(bool).tolist()
                        for name, value in fold_values.items()},
            "bag3_vs_single": _paired(
                labels[held], fold_bag, fold_single, hard_mask[held]),
            "bag3_vs_b0": _paired(
                labels[held], fold_bag, fold_b0, hard_mask[held]),
            "bag3_hard_delta_vs_single": int(
                fold_metrics["bag3"]["hard_correct"] -
                fold_metrics["g3_single"]["hard_correct"]),
            "member_disagreement": disagreements,
            "three_single_copies_equal_single": copy_equal,
            "nonhard_matches_single": True,
            "wall_seconds": float(time.perf_counter() - fold_started),
        }
        folds.append(record)
        _emit(observer, "fold_complete", stage=stage, fold=fold,
              hard_delta=record["bag3_hard_delta_vs_single"],
              wall_seconds=record["wall_seconds"])

    if np.any(coverage != 1):
        raise RuntimeError("outer OOF coverage must be exactly one")
    if any(np.any(value < 0) for value in choices.values()):
        raise RuntimeError("every fixed arm must choose every row exactly once")
    if any(not np.array_equal(value[~hard_mask], choices["g3_single"][~hard_mask])
           for value in choices.values()):
        raise RuntimeError("non-hard OOF choices must match G3-single")

    metrics = {name: _metrics(labels, value, hard_mask)
               for name, value in choices.items()}
    fold_deltas = [record["bag3_hard_delta_vs_single"] for record in folds]
    hard_delta = (metrics["bag3"]["hard_correct"] -
                  metrics["g3_single"]["hard_correct"])
    gate = evaluate_acceptance(
        stage, hard_delta, fold_deltas,
        metrics["bag3"]["overall_correct"],
        metrics["g3_single"]["overall_correct"],
        metrics["b0"]["overall_correct"])
    disagreements = {
        f"member_{left}_vs_member_{right}": _disagreement(
            choices[f"member_{left}"], choices[f"member_{right}"], hard_mask)
        for left, right in ((0, 1), (0, 2), (1, 2))
    }
    result = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "evidence_role": "train_oof_development",
        "stage": stage,
        "fold_seed": int(fold_seed),
        "seed_offset": seed_offset,
        "candidate": "bag3",
        "validation_accessed": False,
        "test_accessed": False,
        "config": FIXED_CONFIG,
        "config_sha256": config_sha256,
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "coverage_exactly_one": True,
        "coverage_counts": coverage.astype(int).tolist(),
        "hard_mask": hard_mask.astype(bool).tolist(),
        "hard_n": int(hard_mask.sum()),
        "overall_n": int(len(bank)),
        "metrics": metrics,
        "choices": {name: value.astype(int).tolist()
                    for name, value in choices.items()},
        "correct": {name: _correct(labels, value).astype(bool).tolist()
                    for name, value in choices.items()},
        "bag3_vs_single": _paired(
            labels, choices["bag3"], choices["g3_single"], hard_mask),
        "bag3_vs_b0": _paired(
            labels, choices["bag3"], choices["b0"], hard_mask),
        "member_disagreement": disagreements,
        "subgroups": _subgroup_report(metadata, labels, hard_mask, choices),
        "controls": {
            "three_single_copies_equal_single": bool(all(copy_controls)),
            "nonhard_matches_single": True,
            "single_and_bag_share_fold_b0_pca_features": True,
            "member_diagnostics_not_selectable": True,
        },
        "folds": folds,
        "gate": gate,
        "gate_passed": bool(gate["passed"]),
        "wall_seconds": float(time.perf_counter() - started),
    }
    _emit(observer, "stage_complete", stage=stage,
          gate_passed=result["gate_passed"],
          wall_seconds=result["wall_seconds"])
    return result
