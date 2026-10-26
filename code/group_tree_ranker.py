"""CatBoost answer-group ranker and deterministic probability fusion."""

from pathlib import Path
import math
import os

import numpy as np


def candidate_group_scores(candidate_scores, valid, groups):
    """Aggregate candidate logits into deterministic answer-group logits."""
    candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    groups = np.asarray(groups, dtype=np.int32)
    if candidate_scores.shape != valid.shape or groups.shape != valid.shape:
        raise ValueError("candidate scores and metadata are not aligned")
    result = np.zeros(candidate_scores.shape, dtype=np.float64)
    group_valid = np.zeros(valid.shape, dtype=bool)
    representatives = np.full(valid.shape, -1, dtype=np.int32)
    for row in range(len(candidate_scores)):
        for slot in sorted(set(groups[row, valid[row]].tolist())):
            members = np.flatnonzero(valid[row] & (groups[row] == slot))
            maximum = float(np.max(candidate_scores[row, members]))
            result[row, slot] = maximum + math.log(float(
                np.exp(candidate_scores[row, members] - maximum).sum()))
            group_valid[row, slot] = True
            representatives[row, slot] = int(members[0])
    return {
        "scores": result, "group_valid": group_valid,
        "representatives": representatives,
    }


def flatten_group_queries(values, group_valid, group_labels=None, query_ids=None):
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(group_valid, dtype=bool)
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("group values and validity mask are not aligned")
    if query_ids is None:
        query_ids = list(range(len(values)))
    if len(query_ids) != len(values):
        raise ValueError("query_ids must align with rows")
    labels = None if group_labels is None else np.asarray(group_labels, dtype=bool)
    if labels is not None and labels.shape != valid.shape:
        raise ValueError("group labels are not aligned")
    rows, groups, targets, slots = [], [], [], []
    for row in range(len(values)):
        for slot in np.flatnonzero(valid[row]):
            rows.append(values[row, slot])
            groups.append(query_ids[row])
            slots.append((row, int(slot)))
            if labels is not None:
                targets.append(float(labels[row, slot]))
    return {
        "values": np.asarray(rows, dtype=np.float32),
        "group_id": np.asarray(groups),
        "labels": None if labels is None else np.asarray(targets, dtype=np.float32),
        "slots": np.asarray(slots, dtype=np.int64).reshape(-1, 2),
    }


def scores_to_group_probabilities(scores, valid):
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    result = np.zeros_like(scores, dtype=np.float64)
    for row in range(len(scores)):
        slots = np.flatnonzero(valid[row])
        if not len(slots):
            continue
        shifted = scores[row, slots] - np.max(scores[row, slots])
        exp = np.exp(shifted)
        result[row, slots] = exp / exp.sum()
    return result


def select_from_group_scores(scores, valid, representatives):
    scores = np.asarray(scores)
    valid = np.asarray(valid, dtype=bool)
    reps = np.asarray(representatives, dtype=np.int32)
    choices = np.zeros(len(scores), dtype=np.int64)
    for row in range(len(scores)):
        slots = np.flatnonzero(valid[row])
        if not len(slots):
            choices[row] = 0
        elif len(slots) == 1:
            choices[row] = reps[row, slots[0]]
        else:
            chosen = max(slots, key=lambda slot: (float(scores[row, slot]),
                                                  -int(reps[row, slot])))
            choices[row] = reps[row, chosen]
    return choices


def mixed_group_choices(b0_scores, tree_scores, valid, representatives, alpha):
    """Fuse group distributions; alpha=0 follows B0 scores mechanically."""
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if alpha == 0.0:
        return select_from_group_scores(b0_scores, valid, representatives)
    b0 = scores_to_group_probabilities(b0_scores, valid)
    tree = scores_to_group_probabilities(tree_scores, valid)
    return select_from_group_scores((1.0 - alpha) * b0 + alpha * tree,
                                    valid, representatives)


def fit_ranker(values, valid, labels, config):
    try:
        from catboost import CatBoostRanker, Pool
    except ImportError as error:
        raise RuntimeError("CatBoost is required for G3 training") from error
    flat = flatten_group_queries(values, valid, labels)
    model = CatBoostRanker(
        loss_function="QuerySoftMax",
        iterations=int(config.get("iterations", 600)),
        depth=int(config["depth"]),
        learning_rate=0.03,
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        border_count=64,
        random_seed=int(config.get("seed", 2028)),
        bootstrap_type="No",
        random_strength=1,
        grow_policy="SymmetricTree",
        boosting_type="Plain",
        thread_count=4,
        allow_writing_files=False,
        verbose=False,
    )
    pool = Pool(flat["values"], label=flat["labels"], group_id=flat["group_id"])
    model.fit(pool)
    return model


def predict_ranker(model, values, valid, ntree_end=None):
    flat = flatten_group_queries(values, valid)
    kwargs = {"thread_count": 4}
    if ntree_end is not None:
        kwargs["ntree_end"] = int(ntree_end)
    prediction = np.asarray(model.predict(flat["values"], **kwargs), dtype=np.float64)
    result = np.zeros(valid.shape, dtype=np.float64)
    for value, (row, slot) in zip(prediction, flat["slots"]):
        result[row, slot] = value
    return result


def save_ranker(model, path):
    """Save one uncompressed JSON ranker asset."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        model.save_model(path.name, format="json")
    finally:
        os.chdir(previous)


def load_ranker(path):
    """Load a JSON ranker without fitting or accessing supervision."""
    try:
        from catboost import CatBoostRanker
    except ImportError as error:
        raise RuntimeError("CatBoost is required for G3 inference") from error
    path = Path(path).resolve()
    model = CatBoostRanker()
    previous = Path.cwd()
    try:
        os.chdir(path.parent)
        model.load_model(path.name, format="json")
    finally:
        os.chdir(previous)
    return model
