"""Train-only fold construction, selection rules, and leakage audits."""

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from features import answer_metadata, extract_features
from group_fusion import (build_group_features, build_group_labels,
                          select_group_choices)
from models import (average_linear_models, fit_scorer, prepare_model_input,
                    score_model, score_prepared)


def _validate_ids_and_strata(ids, strata):
    ids = [str(value) for value in ids]
    strata = [tuple(value) if isinstance(value, (list, tuple, np.ndarray)) else (value,)
              for value in strata]
    if len(ids) != len(strata):
        raise ValueError("ids and strata must have the same length")
    if len(ids) != len(set(ids)):
        raise ValueError("question IDs must be unique")
    return ids, strata


def make_stratified_folds(ids, strata, n_splits=5, seed=2026):
    """Return deterministic folds with exact coverage and balanced strata."""
    ids, strata = _validate_ids_and_strata(ids, strata)
    n_splits = int(n_splits)
    if n_splits < 2 or len(ids) < n_splits:
        raise ValueError("n_splits must be between two and the number of rows")
    rng = np.random.default_rng(int(seed))
    by_stratum = defaultdict(list)
    for index, stratum in enumerate(strata):
        by_stratum[stratum].append(index)

    folds = [[] for _ in range(n_splits)]
    stratum_counts = defaultdict(lambda: [0] * n_splits)
    for stratum in sorted(by_stratum, key=repr):
        members = np.asarray(by_stratum[stratum], dtype=np.int64)
        members = members[rng.permutation(len(members))]
        for member in members:
            counts = stratum_counts[stratum]
            minimum_stratum = min(counts)
            eligible = [fold for fold, count in enumerate(counts)
                        if count == minimum_stratum]
            minimum_total = min(len(folds[fold]) for fold in eligible)
            eligible = [fold for fold in eligible if len(folds[fold]) == minimum_total]
            chosen = eligible[int(rng.integers(len(eligible)))]
            folds[chosen].append(int(member))
            counts[chosen] += 1
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def fusion_fit_stop_split(outer_train_indices, strata, seed, fit_fraction=0.8):
    """Make the fixed deterministic stratified 80/20 fusion split."""
    indices = np.asarray(outer_train_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 2 or len(set(indices.tolist())) != len(indices):
        raise ValueError("outer_train_indices must be unique and contain at least two rows")
    if not 0.0 < float(fit_fraction) < 1.0:
        raise ValueError("fit_fraction must be between zero and one")
    local_strata = [strata[int(index)] for index in indices]
    # Five-way stratification gives the pre-registered 80/20 split.  For tiny
    # fixtures, fall back to a deterministic bounded holdout size.
    if float(fit_fraction) == 0.8 and len(indices) >= 5:
        folds = make_stratified_folds(
            [str(index) for index in indices], local_strata, n_splits=5, seed=seed
        )
        stop_local = max(enumerate(folds), key=lambda item: (len(item[1]), -item[0]))[1]
    else:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(len(indices))
        stop_size = max(1, min(len(indices) - 1,
                               len(indices) - int(np.floor(len(indices) * fit_fraction))))
        stop_local = np.sort(order[:stop_size])
    stop = np.sort(indices[stop_local])
    fit = np.asarray(sorted(set(indices.tolist()) - set(stop.tolist())), dtype=np.int64)
    return fit, stop


def _id_digest(ids):
    payload = json.dumps(sorted(map(str, ids)), ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value):
    """Hash JSON-compatible protocol data without reordering array values."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha256(values):
    """Hash an exact fitted label slice including dtype and shape."""
    array = np.ascontiguousarray(np.asarray(values))
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def audit_fold_ids(ids, outer_heldout, base_fit, fusion_fit, fusion_stop):
    """Validate fold isolation and return stable, non-identifying ID digests."""
    ids = [str(value) for value in ids]
    if len(ids) != len(set(ids)):
        raise ValueError("question IDs must be unique")
    named = {
        "outer_heldout": np.asarray(outer_heldout, dtype=np.int64),
        "base_fit": np.asarray(base_fit, dtype=np.int64),
        "fusion_fit": np.asarray(fusion_fit, dtype=np.int64),
        "fusion_stop": np.asarray(fusion_stop, dtype=np.int64),
    }
    sets = {}
    for name, values in named.items():
        if values.ndim != 1 or np.any(values < 0) or np.any(values >= len(ids)):
            raise ValueError(f"{name} contains invalid row indices")
        sets[name] = set(values.tolist())
    held = sets["outer_heldout"]
    if any(held & sets[name] for name in ("base_fit", "fusion_fit", "fusion_stop")):
        raise ValueError("outer-held-out IDs intersect a fold-internal fit set")
    if sets["fusion_fit"] & sets["fusion_stop"]:
        raise ValueError("fusion fit and stop IDs must be disjoint")
    if sets["fusion_fit"] | sets["fusion_stop"] != sets["base_fit"]:
        raise ValueError("fusion fit/stop must partition base-fit IDs")
    return {
        "counts": {name: len(values) for name, values in sets.items()},
        "sha256": {
            name: _id_digest(ids[index] for index in sorted(values))
            for name, values in sets.items()
        },
    }


def select_candidate_epoch(rows):
    """Maximize overall then hard correctness; ties use the smaller epoch."""
    rows = list(rows)
    if not rows:
        raise ValueError("candidate epoch selection requires at least one row")
    best = max(rows, key=lambda row: (
        int(row["overall_correct"]), int(row["hard_correct"]), -int(row["epoch"])
    ))
    return int(best["epoch"])


def select_fusion_setting(rows):
    """Apply the fixed fusion metric, L2, and epoch tie-break ordering."""
    rows = list(rows)
    if not rows:
        raise ValueError("fusion selection requires at least one row")
    allowed = {0.001, 0.01}
    if any(float(row["l2"]) not in allowed for row in rows):
        raise ValueError("fusion L2 must be one of 0.001 or 0.01")
    best = max(rows, key=lambda row: (
        int(row["overall_correct"]), int(row["hard_correct"]),
        float(row["l2"]), -int(row["epoch"]),
    ))
    return {"l2": float(best["l2"]), "epoch": int(best["epoch"])}


def _lower_median(values):
    values = sorted(values)
    if not values:
        raise ValueError("lower median requires at least one value")
    return values[(len(values) - 1) // 2]


def select_final_refit_parameters(fold_rows):
    """Derive frozen full-train parameters from the five outer-fold choices."""
    rows = list(fold_rows)
    if not rows:
        raise ValueError("final refit selection requires fold rows")
    candidate_epoch = int(_lower_median([int(row["candidate_epoch"]) for row in rows]))
    l2_counts = Counter(float(row["fusion_l2"]) for row in rows)
    highest_count = max(l2_counts.values())
    fusion_l2 = max(value for value, count in l2_counts.items() if count == highest_count)
    matching_epochs = [int(row["fusion_epoch"]) for row in rows
                       if float(row["fusion_l2"]) == fusion_l2]
    if not matching_epochs:
        matching_epochs = [int(row["fusion_epoch"]) for row in rows]
    return {
        "candidate_epoch": candidate_epoch,
        "fusion_l2": float(fusion_l2),
        "fusion_epoch": int(_lower_median(matching_epochs)),
    }


def _strict_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def validate_development_config(config, row_count=None):
    """Apply the pre-registered protocol, allowing bounded synthetic smoke overrides."""
    config = dict(config)
    synthetic = bool(config.get("synthetic", False))
    defaults = {
        "candidate_max_epochs": 150,
        "fusion_max_epochs": 100,
        "outer_splits": 5,
        "inner_splits": 4,
        "candidate_seeds": [2028],
        "fusion_l2_values": [0.001, 0.01],
        "fold_seed": 2026,
    }
    resolved = {**defaults, **config}
    seeds = [_strict_integer(value, "candidate seed") for value in resolved["candidate_seeds"]]
    candidate_max_epochs = _strict_integer(resolved["candidate_max_epochs"],
                                           "candidate_max_epochs")
    fusion_max_epochs = _strict_integer(resolved["fusion_max_epochs"],
                                        "fusion_max_epochs")
    fold_seed = _strict_integer(resolved["fold_seed"], "fold_seed")
    if not 1 <= candidate_max_epochs <= 150 or not 1 <= fusion_max_epochs <= 100:
        raise ValueError("candidate/fusion epoch limits must be within 1..150 and 1..100")
    if not synthetic:
        if candidate_max_epochs != 150:
            raise ValueError("formal candidate_max_epochs is fixed at 150")
        if fusion_max_epochs != 100:
            raise ValueError("formal fusion_max_epochs is fixed at 100")
        if seeds not in ([2028], [2026, 2027, 2028]):
            raise ValueError("formal candidate seeds are fixed")
        if int(resolved["outer_splits"]) != 5 or int(resolved["inner_splits"]) != 4:
            raise ValueError("formal fold counts are fixed at five outer/four inner")
        if fold_seed not in {2026, 2027, 2028}:
            raise ValueError("formal fold_seed must be 2026, 2027, or 2028")
        if resolved.get("smoke_fixture") or "smoke_row_limit" in resolved:
            raise ValueError("formal runs cannot declare smoke overrides")
    else:
        if resolved.get("smoke_fixture") is not True:
            raise ValueError("synthetic override requires an explicit smoke fixture")
        limit = _strict_integer(resolved.get("smoke_row_limit"), "smoke_row_limit")
        if not 1 <= limit <= 30:
            raise ValueError("smoke_row_limit must be between 1 and 30")
        if row_count is None or not 1 <= int(row_count) <= limit:
            raise ValueError("smoke override exceeds its explicit row cap")
        if not seeds:
            raise ValueError("synthetic smoke seeds must be nonempty")
    if [float(value) for value in resolved["fusion_l2_values"]] != [0.001, 0.01]:
        raise ValueError("fusion L2 values are fixed at 0.001 and 0.01")
    resolved["candidate_seeds"] = seeds
    resolved["candidate_max_epochs"] = candidate_max_epochs
    resolved["fusion_max_epochs"] = fusion_max_epochs
    resolved["fold_seed"] = fold_seed
    return resolved


def validate_selected_parameters(selected):
    """Validate the only candidate/fusion values permitted by the fixed protocol."""
    selected = dict(selected)
    required = {"candidate_epoch", "fusion_l2", "fusion_epoch"}
    if set(selected) != required:
        raise ValueError("selected parameters must contain exactly the three frozen values")
    candidate_epoch = _strict_integer(selected["candidate_epoch"], "candidate_epoch")
    fusion_epoch = _strict_integer(selected["fusion_epoch"], "fusion_epoch")
    fusion_l2 = float(selected["fusion_l2"])
    if not 1 <= candidate_epoch <= 150:
        raise ValueError("candidate_epoch must be within 1..150")
    if not 1 <= fusion_epoch <= 100:
        raise ValueError("fusion_epoch must be within 1..100")
    if fusion_l2 not in {0.001, 0.01}:
        raise ValueError("fusion_l2 must be 0.001 or 0.01")
    return {"candidate_epoch": candidate_epoch, "fusion_l2": fusion_l2,
            "fusion_epoch": fusion_epoch}


def build_formal_config_binding(config, feature_names):
    """Return the train-OOF settings that a formal final fit must reproduce."""
    return {
        "candidate_seeds": [int(value) for value in config["candidate_seeds"]],
        "candidate_max_epochs": int(config["candidate_max_epochs"]),
        "fusion_max_epochs": int(config["fusion_max_epochs"]),
        "fusion_l2_values": [float(value) for value in config["fusion_l2_values"]],
        "outer_splits": int(config["outer_splits"]),
        "inner_splits": int(config["inner_splits"]),
        "fold_seed": int(config["fold_seed"]),
        "feature_mode": str(config.get("feature_mode", "embedding_text_verifier")),
        "exclude_features": sorted(str(value) for value in config.get("exclude_features", [])),
        "feature_names": [str(value) for value in feature_names],
    }


def load_selected_parameters(config, synthetic, bank_ids=None, feature_names=None):
    """Load synthetic smoke values directly or formal values from a hashed OOF artifact."""
    if synthetic:
        selected = validate_selected_parameters(config.get("selected_parameters", {}))
        return selected, {"source": "explicit_small_smoke_fixture"}
    if "selected_parameters" in config:
        raise ValueError("formal fit-final parameters must come from a hashed OOF artifact")
    if bank_ids is None or feature_names is None:
        raise ValueError("formal fit-final requires current train IDs and feature contract")
    bank_ids = [str(value) for value in bank_ids]
    config = validate_development_config(config, row_count=len(bank_ids))
    path_value = config.get("oof_selection_artifact")
    expected = str(config.get("oof_selection_sha256", "")).lower()
    if not path_value or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("formal fit-final requires an OOF artifact path and SHA-256")
    path = Path(path_value)
    encoded = path.read_bytes()
    actual = hashlib.sha256(encoded).hexdigest()
    if actual != expected:
        raise ValueError("OOF selection artifact SHA-256 mismatch")
    payload = json.loads(encoded.decode("utf-8"))
    if payload.get("evidence_role") != "development_selection":
        raise ValueError("OOF selection artifact has the wrong evidence role")
    if len(bank_ids) != len(set(bank_ids)):
        raise ValueError("current train IDs must be unique")
    if payload.get("fold_seed") != int(config["fold_seed"]):
        raise ValueError("OOF selection artifact fold seed does not match final-fit config")
    if payload.get("train_row_count") != len(bank_ids):
        raise ValueError("OOF selection artifact coverage row count does not match train")
    if payload.get("train_ids_sha256") != canonical_sha256(bank_ids):
        raise ValueError("OOF selection artifact train IDs do not match current train order")
    stored_config = payload.get("formal_config")
    stored_digest = payload.get("formal_config_sha256")
    if (not isinstance(stored_config, dict) or
            stored_digest != canonical_sha256(stored_config)):
        raise ValueError("OOF selection artifact config digest is invalid")
    current_config = build_formal_config_binding(config, feature_names)
    if stored_config != current_config:
        raise ValueError("OOF selection artifact formal config does not match final fit")
    folds = payload.get("folds")
    coverage = payload.get("coverage_counts")
    if (not isinstance(coverage, list) or len(coverage) != len(bank_ids) or
            any(type(value) is not int or value != 1 for value in coverage)):
        raise ValueError("OOF selection artifact coverage does not match current train")
    if not isinstance(folds, list) or len(folds) != 5:
        raise ValueError("OOF selection artifact lacks complete five-fold coverage")
    if (any(not isinstance(fold, dict) for fold in folds) or
            sorted(fold.get("outer_fold") for fold in folds) != list(range(5))):
        raise ValueError("OOF selection artifact outer-heldout partition has invalid fold IDs")
    partition_counts = np.zeros(len(bank_ids), dtype=np.int16)
    for fold in folds:
        heldout = fold.get("outer_heldout_indices")
        if (not isinstance(heldout, list) or not heldout or
                any(type(index) is not int or index < 0 or index >= len(bank_ids)
                    for index in heldout) or len(heldout) != len(set(heldout))):
            raise ValueError("OOF selection artifact outer-heldout partition is invalid")
        partition_counts[heldout] += 1
    if partition_counts.tolist() != coverage:
        raise ValueError("OOF selection artifact outer-heldout partition is incomplete")
    derived = validate_selected_parameters(select_final_refit_parameters(folds))
    declared = validate_selected_parameters(payload.get("final_refit_parameters", {}))
    if declared != derived:
        raise ValueError("OOF selection artifact parameters do not match its fold selections")
    return declared, {"source": str(path), "sha256": actual}


def _select_feature_columns(features, names, config):
    excluded = set(config.get("exclude_features", []))
    keep = [index for index, name in enumerate(names)
            if not any(token in name for token in excluded)]
    return features[:, :, keep], [names[index] for index in keep]


def _development_strata(labels, hard_mask, metadata):
    majority_correct = np.zeros(len(labels), dtype=np.bool_)
    for row in range(len(labels)):
        available = np.flatnonzero(metadata["valid"][row])
        if len(available):
            votes = metadata["votes"][row, available]
            choice = int(available[int(np.argmax(votes))])
            majority_correct[row] = labels[row, choice]
    return [(bool(hard_mask[row]), bool(majority_correct[row]),
             bool(labels[row].sum() > 1)) for row in range(len(labels))]


def _fixed_group_choices(scores, valid, groups):
    choices = []
    for row in range(len(scores)):
        available = np.flatnonzero(valid[row])
        if not len(available):
            choices.append(0)
            continue
        decisions = []
        for slot in sorted(set(groups[row, available].tolist())):
            members = available[groups[row, available] == slot]
            values = scores[row, members].astype(np.float64)
            maximum = float(values.max())
            # logmeanexp + log(size) is exactly logsumexp.
            decision = maximum + math.log(float(np.exp(values - maximum).sum()))
            decisions.append((decision, -int(members[0]), int(members[0])))
        choices.append(max(decisions)[2])
    return choices


def _correct_counts(choices, labels, hard_mask):
    choices = np.asarray(choices, dtype=np.int64)
    correct = labels[np.arange(len(labels)), choices]
    return int(correct.sum()), int(correct[np.asarray(hard_mask, dtype=np.bool_)].sum())


def _notify(config, outer_fold, decision, row_indices, label_slice, label_kind,
            hard_mask_slice=None):
    observer = config.get("selection_observer")
    if observer is not None:
        event = {
            "outer_fold": int(outer_fold),
            "decision": str(decision),
            "row_indices": [int(value) for value in row_indices],
            "label_kind": str(label_kind),
            "label_shape": list(np.asarray(label_slice).shape),
            "labels_sha256": array_sha256(label_slice),
        }
        if hard_mask_slice is not None:
            event["hard_mask_shape"] = list(np.asarray(hard_mask_slice).shape)
            event["hard_mask_sha256"] = array_sha256(hard_mask_slice)
        observer(event)


def _candidate_inner_oof(ids, x, labels, valid, metadata, hard_mask, outer_train,
                         strata, config, outer_fold):
    max_epochs = int(config["candidate_max_epochs"])
    inner_folds = make_stratified_folds(
        [str(index) for index in outer_train], [strata[index] for index in outer_train],
        n_splits=int(config["inner_splits"]), seed=2026 + outer_fold,
    )
    epoch_scores = {epoch: np.zeros((len(labels), 4), dtype=np.float32)
                    for epoch in range(1, max_epochs + 1)}
    coverage = np.zeros(len(labels), dtype=np.int16)
    fit_audits = []
    for inner_held_local in inner_folds:
        inner_held = outer_train[inner_held_local]
        inner_fit = np.asarray(sorted(set(outer_train.tolist()) - set(inner_held.tolist())),
                               dtype=np.int64)
        _notify(config, outer_fold, "candidate_inner_fit", inner_fit,
                labels[inner_fit], "candidate")
        fit_audits.append({
            "count": int(len(inner_fit)),
            "sha256": _id_digest(ids[index] for index in inner_fit),
        })
        for seed in config["candidate_seeds"]:
            prepared_heldout = None

            def monitor(epoch, snapshot, _info, held=inner_held):
                nonlocal prepared_heldout
                if prepared_heldout is None:
                    prepared_heldout = prepare_model_input(x[held], snapshot)
                epoch_scores[epoch][held] += score_prepared(
                    prepared_heldout, snapshot,
                ) / len(config["candidate_seeds"])

            fit_scorer(x[inner_fit], labels[inner_fit], valid[inner_fit], {
                "architecture": "linear", "objective": "listwise",
                "learning_rate": 0.003, "l2": 0.0001, "batch_size": 128,
                "epochs": max_epochs, "seed": int(seed), "feature_names": config["feature_names"],
                "monitor_training_loss": False,
            }, monitor=monitor)
        coverage[inner_held] += 1
    if np.any(coverage[outer_train] != 1):
        raise RuntimeError("inner OOF coverage must be exactly one")
    rows = []
    for epoch, scores in epoch_scores.items():
        choices = _fixed_group_choices(scores[outer_train], valid[outer_train],
                                       metadata["group_index"][outer_train])
        overall, hard = _correct_counts(choices, labels[outer_train], hard_mask[outer_train])
        rows.append({"epoch": epoch, "overall_correct": overall, "hard_correct": hard})
    return epoch_scores, select_candidate_epoch(rows), rows, fit_audits


def _fit_candidate_models(x, labels, valid, fit_indices, score_indices, epoch,
                          config, outer_fold, decision):
    _notify(config, outer_fold, decision, fit_indices, labels[fit_indices], "candidate")
    models, scores = [], []
    for seed in config["candidate_seeds"]:
        model = fit_scorer(x[fit_indices], labels[fit_indices], valid[fit_indices], {
            "architecture": "linear", "objective": "listwise",
            "learning_rate": 0.003, "l2": 0.0001, "batch_size": 128,
            "epochs": int(epoch), "seed": int(seed), "feature_names": config["feature_names"],
        })
        models.append(model)
        scores.append(score_model(x[score_indices], model))
    return models, np.mean(scores, axis=0, dtype=np.float32)


def run_train_oof_development(bank, embeddings, labels, hard_mask, config):
    """Run nested train-only OOF selection and score every outer row once."""
    started = time.perf_counter()
    config = validate_development_config(config, row_count=len(bank))
    labels = np.asarray(labels, dtype=np.bool_)
    hard_mask = np.asarray(hard_mask, dtype=np.bool_)
    embeddings = np.asarray(embeddings)
    ids = [str(row["id"]) for row in bank]
    if (labels.shape != (len(bank), 4) or hard_mask.shape != (len(bank),) or
            embeddings.ndim != 3 or embeddings.shape[:2] != (len(bank), 4)):
        raise ValueError("train inputs, labels, and masks are not aligned")
    metadata = answer_metadata(bank)
    if np.any(labels & ~metadata["valid"]) or np.any(~labels.any(axis=1)):
        raise ValueError("every train row needs a parseable correct candidate")
    candidate_x, feature_names = extract_features(
        bank, embeddings, config.get("feature_mode", "embedding_text_verifier")
    )
    candidate_x, feature_names = _select_feature_columns(candidate_x, feature_names, config)
    config["feature_names"] = feature_names
    formal_config = build_formal_config_binding(config, feature_names)
    strata = _development_strata(labels, hard_mask, metadata)
    outer_folds = make_stratified_folds(
        ids, strata, n_splits=int(config["outer_splits"]), seed=int(config.get("fold_seed", 2026))
    )
    oof_choices = np.full(len(bank), -1, dtype=np.int32)
    coverage = np.zeros(len(bank), dtype=np.int16)
    fold_rows = []
    all_indices = set(range(len(bank)))

    for outer_fold, outer_held in enumerate(outer_folds):
        outer_train = np.asarray(sorted(all_indices - set(outer_held.tolist())), dtype=np.int64)
        inner_scores, candidate_epoch, candidate_metrics, candidate_inner_audits = _candidate_inner_oof(
            ids, candidate_x, labels, metadata["valid"], metadata, hard_mask,
            outer_train, strata, config, outer_fold,
        )
        _, outer_candidate_scores = _fit_candidate_models(
            candidate_x, labels, metadata["valid"], outer_train, outer_held,
            candidate_epoch, config, outer_fold, "candidate_refit",
        )
        candidate_oof_scores = inner_scores[candidate_epoch]
        group_x, group_names, group_valid, representatives = build_group_features(
            candidate_oof_scores, (candidate_x, feature_names), embeddings, metadata
        )
        group_labels, labels_valid = build_group_labels(labels, metadata)
        if not np.array_equal(group_valid, labels_valid):
            raise RuntimeError("group validity mismatch")
        fusion_fit, fusion_stop = fusion_fit_stop_split(
            outer_train, strata, seed=2600 + outer_fold
        )
        _notify(config, outer_fold, "fusion_stop_selection", fusion_stop,
                labels[fusion_stop], "candidate", hard_mask[fusion_stop])
        _notify(config, outer_fold, "fusion_fit", fusion_fit,
                group_labels[fusion_fit], "group")
        fusion_rows = []
        for l2 in config["fusion_l2_values"]:
            stale = 0
            best_key = (-1, -1)
            prepared_stop = None

            def monitor(epoch, snapshot, _info, current_l2=float(l2)):
                nonlocal stale, best_key, prepared_stop
                if prepared_stop is None:
                    prepared_stop = prepare_model_input(group_x[fusion_stop], snapshot)
                stop_scores = score_prepared(prepared_stop, snapshot)
                choices = select_group_choices(
                    stop_scores, group_valid[fusion_stop], representatives[fusion_stop]
                )
                overall, hard = _correct_counts(
                    choices, labels[fusion_stop], hard_mask[fusion_stop]
                )
                fusion_rows.append({"l2": current_l2, "epoch": int(epoch),
                                    "overall_correct": overall, "hard_correct": hard})
                key = (overall, hard)
                if key > best_key:
                    best_key, stale = key, 0
                else:
                    stale += 1
                return stale < 10

            fit_scorer(group_x[fusion_fit], group_labels[fusion_fit], group_valid[fusion_fit], {
                "architecture": "linear", "objective": "listwise",
                "learning_rate": 0.003, "l2": float(l2), "batch_size": 128,
                "epochs": int(config["fusion_max_epochs"]), "seed": 2600 + outer_fold,
                "feature_names": group_names,
                "monitor_training_loss": False,
            }, monitor=monitor)
        fusion_setting = select_fusion_setting(fusion_rows)
        _notify(config, outer_fold, "fusion_refit", outer_train,
                group_labels[outer_train], "group")
        fusion_model = fit_scorer(
            group_x[outer_train], group_labels[outer_train], group_valid[outer_train], {
                "architecture": "linear", "objective": "listwise",
                "learning_rate": 0.003, "l2": fusion_setting["l2"], "batch_size": 128,
                "epochs": fusion_setting["epoch"], "seed": 2600 + outer_fold,
                "feature_names": group_names,
            }
        )
        outer_score_matrix = np.zeros((len(bank), 4), dtype=np.float32)
        outer_score_matrix[outer_held] = outer_candidate_scores
        outer_group_x, _, outer_group_valid, outer_reps = build_group_features(
            outer_score_matrix, (candidate_x, feature_names), embeddings, metadata
        )
        outer_group_scores = score_model(outer_group_x[outer_held], fusion_model)
        choices = select_group_choices(
            outer_group_scores, outer_group_valid[outer_held], outer_reps[outer_held]
        )
        oof_choices[outer_held] = choices
        coverage[outer_held] += 1
        id_audit = audit_fold_ids(ids, outer_held, outer_train, fusion_fit, fusion_stop)
        fold_rows.append({
            "outer_fold": outer_fold,
            "outer_heldout_indices": outer_held.tolist(),
            "base_fit_indices": outer_train.tolist(),
            "fusion_fit_indices": fusion_fit.tolist(),
            "fusion_stop_indices": fusion_stop.tolist(),
            "candidate_epoch": candidate_epoch,
            "candidate_epoch_metrics": candidate_metrics,
            "fusion_l2": fusion_setting["l2"],
            "fusion_epoch": fusion_setting["epoch"],
            "id_audit": id_audit,
            "scale_fit_audits": {
                "candidate_inner": candidate_inner_audits,
                "candidate_refit": {
                    "count": int(len(outer_train)),
                    "sha256": id_audit["sha256"]["base_fit"],
                },
                "fusion_search": {
                    "count": int(len(fusion_fit)),
                    "sha256": id_audit["sha256"]["fusion_fit"],
                },
                "fusion_refit": {
                    "count": int(len(outer_train)),
                    "sha256": id_audit["sha256"]["base_fit"],
                },
            },
        })

    if np.any(coverage != 1) or np.any(oof_choices < 0):
        raise RuntimeError("outer OOF coverage must be exactly one")
    correct = labels[np.arange(len(labels)), oof_choices]
    final_parameters = select_final_refit_parameters(fold_rows)
    return {
        "evidence_role": "development_selection",
        "fold_seed": int(config.get("fold_seed", 2026)),
        "train_row_count": len(ids),
        "train_ids_sha256": canonical_sha256(ids),
        "formal_config": formal_config,
        "formal_config_sha256": canonical_sha256(formal_config),
        "choices": oof_choices.astype(int).tolist(),
        "correct_mask": correct.astype(bool).tolist(),
        "coverage_counts": coverage.astype(int).tolist(),
        "folds": fold_rows,
        "final_refit_parameters": final_parameters,
        "costs": {"wall_seconds": float(time.perf_counter() - started)},
    }


def fit_final_from_oof(bank, embeddings, labels, hard_mask, config):
    """Regenerate frozen full-train OOF scores, then fit fusion and candidate models."""
    config = validate_development_config(config, row_count=len(bank))
    labels = np.asarray(labels, dtype=np.bool_)
    hard_mask = np.asarray(hard_mask, dtype=np.bool_)
    embeddings = np.asarray(embeddings)
    metadata = answer_metadata(bank)
    x, names = extract_features(bank, embeddings, config.get("feature_mode", "embedding_text_verifier"))
    x, names = _select_feature_columns(x, names, config)
    config["feature_names"] = names
    ids = [str(row["id"]) for row in bank]
    selected, selection_provenance = load_selected_parameters(
        config, synthetic=bool(config.get("synthetic", False)), bank_ids=ids,
        feature_names=names,
    )
    strata = _development_strata(labels, hard_mask, metadata)
    folds = make_stratified_folds(ids, strata, n_splits=5,
                                  seed=int(config.get("fold_seed", 2026)))
    candidate_oof = np.zeros((len(bank), 4), dtype=np.float32)
    all_indices = set(range(len(bank)))
    for held in folds:
        fit = np.asarray(sorted(all_indices - set(held.tolist())), dtype=np.int64)
        _, candidate_oof[held] = _fit_candidate_models(
            x, labels, metadata["valid"], fit, held, selected["candidate_epoch"],
            config, -1, "final_candidate_oof_fit",
        )
    group_x, group_names, group_valid, _ = build_group_features(
        candidate_oof, (x, names), embeddings, metadata
    )
    group_labels, _ = build_group_labels(labels, metadata)
    fusion_model = fit_scorer(group_x, group_labels, group_valid, {
        "architecture": "linear", "objective": "listwise", "learning_rate": 0.003,
        "l2": float(selected["fusion_l2"]), "batch_size": 128,
        "epochs": int(selected["fusion_epoch"]), "seed": 2026,
        "feature_names": group_names,
    })
    full = np.arange(len(bank), dtype=np.int64)
    candidate_models, _ = _fit_candidate_models(
        x, labels, metadata["valid"], full, full, selected["candidate_epoch"],
        config, -1, "final_candidate_full_fit",
    )
    candidate_model = (average_linear_models(candidate_models)
                       if len(candidate_models) > 1 else candidate_models[0])
    return {"candidate_model": candidate_model, "fusion_model": fusion_model,
            "parameters": selected, "selection_provenance": selection_provenance,
            "evidence_role": "development_selection"}

