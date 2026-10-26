"""Pure query-bootstrap and probability-fusion helpers for BAG3."""

import numpy as np

from group_tree_ranker import (scores_to_group_probabilities,
                               select_from_group_scores)


__all__ = [
    "stratified_query_bootstrap",
    "mean_group_probabilities",
    "bagged_group_choices",
]


def _validated_strata(strata):
    try:
        values = np.asarray(strata)
    except (TypeError, ValueError) as error:
        raise ValueError("strata must be an N-by-2 integer array") from error
    if values.size == 0 and values.shape in ((0,), (0, 2)):
        return np.empty((0, 2), dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("strata must be an N-by-2 integer array")
    if values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise ValueError("strata values must be integers")
    result = values.astype(np.int64, copy=False)
    if np.any(result < 0):
        raise ValueError("strata values must be non-negative")
    return result


def _validated_valid_mask(valid, row_count):
    mask = np.asarray(valid)
    if mask.shape != (row_count, 4):
        raise ValueError("valid must have shape [N, 4]")
    if mask.dtype.kind != "b":
        raise ValueError("valid must be a boolean mask")
    return mask


def stratified_query_bootstrap(strata, eligible_indices, seed):
    """Draw eligible input-row indices with replacement within each stratum."""
    strata_array = _validated_strata(strata)
    eligible = np.asarray(eligible_indices)
    if eligible.ndim != 1:
        raise ValueError("eligible_indices must be one-dimensional")
    if eligible.size == 0:
        return np.empty(0, dtype=np.int64)
    if eligible.dtype.kind not in "iu" or eligible.dtype.kind == "b":
        raise ValueError("eligible_indices must contain integers")
    eligible = eligible.astype(np.int64, copy=False)
    if np.any(eligible < 0) or np.any(eligible >= len(strata_array)):
        raise ValueError("eligible_indices contain an out-of-range row")
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError("eligible_indices must not contain duplicates")

    by_stratum = {}
    for index in eligible:
        key = tuple(int(value) for value in strata_array[index])
        by_stratum.setdefault(key, []).append(int(index))

    rng = np.random.default_rng(seed)
    draws = [
        rng.choice(np.asarray(indices, dtype=np.int64),
                   size=len(indices), replace=True)
        for indices in by_stratum.values()
    ]
    return np.concatenate(draws).astype(np.int64, copy=False)


def mean_group_probabilities(member_scores, valid):
    """Softmax each member over valid groups, then average its probabilities."""
    try:
        scores = np.asarray(member_scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("member_scores must be a finite [M, N, 4] array") from error
    if scores.ndim != 3 or scores.shape[0] < 1 or scores.shape[2] != 4:
        raise ValueError("member_scores must have shape [M, N, 4] with M >= 1")
    if not np.isfinite(scores).all():
        raise ValueError("member_scores must contain only finite values")
    mask = _validated_valid_mask(valid, scores.shape[1])

    probabilities = [
        scores_to_group_probabilities(member, mask) for member in scores
    ]
    if all(np.array_equal(scores[0], member) for member in scores[1:]):
        return probabilities[0]
    return np.mean(np.stack(probabilities, axis=0), axis=0, dtype=np.float64)


def bagged_group_choices(b0_scores, member_scores, valid, representatives,
                         alpha=0.5):
    """Fuse B0 with the equal-weight member probability mean and select rows."""
    try:
        b0 = np.asarray(b0_scores, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("b0_scores must be a finite [N, 4] array") from error
    if b0.ndim != 2 or b0.shape[1:] != (4,):
        raise ValueError("b0_scores must have shape [N, 4]")
    if not np.isfinite(b0).all():
        raise ValueError("b0_scores must contain only finite values")
    mask = _validated_valid_mask(valid, len(b0))

    reps = np.asarray(representatives)
    if reps.shape != b0.shape:
        raise ValueError("representatives must have shape [N, 4]")
    if reps.dtype.kind not in "iu" or reps.dtype.kind == "b":
        raise ValueError("representatives must contain integers")
    reps = reps.astype(np.int64, copy=False)
    if np.any((reps[mask] < 0) | (reps[mask] >= 4)):
        raise ValueError("valid groups must identify an existing candidate")

    try:
        weight = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must lie in [0, 1]") from error
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    member_probabilities = mean_group_probabilities(member_scores, mask)
    if member_probabilities.shape != b0.shape:
        raise ValueError("member_scores and b0_scores are not aligned")
    if weight == 0.0:
        return select_from_group_scores(b0, mask, reps)
    if weight == 1.0:
        return select_from_group_scores(member_probabilities, mask, reps)
    b0_probabilities = scores_to_group_probabilities(b0, mask)
    combined = ((1.0 - weight) * b0_probabilities
                + weight * member_probabilities)
    return select_from_group_scores(combined, mask, reps)
