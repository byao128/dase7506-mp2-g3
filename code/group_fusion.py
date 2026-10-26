"""Fixed answer-group features for the single linear listwise fusion model."""

import math

import numpy as np


GROUP_FEATURE_NAMES = [
    "group.base_logmeanexp",
    "group.log_size",
    "group.parseable_share",
    "group.mean_arithmetic_pass",
    "group.max_arithmetic_pass",
    "group.mean_arithmetic_fail",
    "group.max_arithmetic_fail",
    "group.mean_final_support",
    "group.max_final_support",
    "group.mean_pairwise_cosine_distance",
    "group.singleton",
    "interaction.log_size_x_diversity",
    "interaction.log_size_x_pass",
    "interaction.log_size_x_fail",
    "interaction.log_size_x_base_score",
]


def _candidate_feature_input(candidate_features, metadata):
    if isinstance(candidate_features, tuple) and len(candidate_features) == 2:
        values, names = candidate_features
    elif isinstance(candidate_features, dict):
        values = candidate_features["values"]
        names = candidate_features["feature_names"]
    else:
        values = candidate_features
        names = metadata.get("candidate_feature_names")
    values = np.asarray(values)
    if names is None:
        raise ValueError("candidate feature names are required")
    names = list(names)
    if values.ndim != 3 or values.shape[1] != 4 or values.shape[2] != len(names):
        raise ValueError("candidate features must have shape [N, 4, D]")
    if not np.isfinite(values).all():
        raise ValueError("candidate features must be finite")
    required = {
        "pass": "verifier.pass_ratio",
        "fail": "verifier.fail_ratio",
        "support": "verifier.any_checked_matches_final",
    }
    try:
        columns = {key: names.index(name) for key, name in required.items()}
    except ValueError as error:
        raise ValueError("candidate features lack fixed verifier columns") from error
    return values.astype(np.float32, copy=False), columns


def _metadata_arrays(metadata, n):
    valid = np.asarray(metadata["valid"], dtype=np.bool_)
    groups = np.asarray(metadata["group_index"], dtype=np.int32)
    if valid.shape != (n, 4) or groups.shape != (n, 4):
        raise ValueError("metadata must contain aligned [N, 4] valid/group arrays")
    if np.any((groups < -1) | (groups > 3)) or np.any(valid != (groups >= 0)):
        raise ValueError("group metadata is inconsistent")
    return valid, groups


def _logmeanexp(values):
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values.astype(np.float64) - maximum).mean()))


def _mean_pairwise_cosine_distance(vectors):
    if len(vectors) < 2:
        return 0.0
    vectors = vectors.astype(np.float64, copy=False)
    norms = np.linalg.norm(vectors, axis=1)
    distances = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            denominator = norms[left] * norms[right]
            similarity = (float(np.dot(vectors[left], vectors[right]) / denominator)
                          if denominator > 0 else 1.0)
            distances.append(1.0 - max(-1.0, min(1.0, similarity)))
    return float(np.mean(distances))


def build_group_features(mean_candidate_scores, candidate_features, embeddings, metadata):
    """Aggregate mean candidate logits into four deterministic answer-group slots."""
    scores = np.asarray(mean_candidate_scores)
    embeddings = np.asarray(embeddings)
    if scores.ndim != 2 or scores.shape[1] != 4 or not np.isfinite(scores).all():
        raise ValueError("mean candidate scores must be finite [N, 4]")
    if (embeddings.ndim != 3 or embeddings.shape[:2] != scores.shape or
            not np.issubdtype(embeddings.dtype, np.number) or
            np.issubdtype(embeddings.dtype, np.complexfloating) or
            not np.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite real [N, 4, D]")
    candidate, columns = _candidate_feature_input(candidate_features, metadata)
    if candidate.shape[:2] != scores.shape:
        raise ValueError("candidate features and scores are not aligned")
    valid_candidates, groups = _metadata_arrays(metadata, len(scores))

    result = np.zeros((len(scores), 4, len(GROUP_FEATURE_NAMES)), dtype=np.float32)
    group_valid = np.zeros((len(scores), 4), dtype=np.bool_)
    representatives = np.full((len(scores), 4), -1, dtype=np.int32)
    for row in range(len(scores)):
        parseable_count = int(valid_candidates[row].sum())
        for slot in range(4):
            members = np.flatnonzero(valid_candidates[row] & (groups[row] == slot))
            if not len(members):
                continue
            group_valid[row, slot] = True
            representatives[row, slot] = int(members[0])
            size = len(members)
            base = _logmeanexp(scores[row, members])
            log_size = math.log(float(size))
            passed = candidate[row, members, columns["pass"]]
            failed = candidate[row, members, columns["fail"]]
            support = candidate[row, members, columns["support"]]
            diversity = _mean_pairwise_cosine_distance(embeddings[row, members])
            mean_pass = float(passed.mean())
            mean_fail = float(failed.mean())
            result[row, slot] = [
                base, log_size, size / float(parseable_count),
                mean_pass, float(passed.max()), mean_fail, float(failed.max()),
                float(support.mean()), float(support.max()), diversity,
                float(size == 1),
                log_size * diversity, log_size * mean_pass,
                log_size * mean_fail, log_size * base,
            ]
    if not np.isfinite(result).all():
        raise ValueError("group features must be finite")
    return result, GROUP_FEATURE_NAMES.copy(), group_valid, representatives


def build_group_labels(candidate_labels, metadata):
    """Map any correct candidate onto its parse-equivalent group slot."""
    labels = np.asarray(candidate_labels, dtype=np.bool_)
    if labels.ndim != 2 or labels.shape[1] != 4:
        raise ValueError("candidate labels must have shape [N, 4]")
    valid, groups = _metadata_arrays(metadata, len(labels))
    if np.any(labels & ~valid):
        raise ValueError("correct labels must identify parseable candidates")
    group_labels = np.zeros((len(labels), 4), dtype=np.bool_)
    group_valid = np.zeros((len(labels), 4), dtype=np.bool_)
    for row in range(len(labels)):
        for slot in range(4):
            members = valid[row] & (groups[row] == slot)
            group_valid[row, slot] = bool(members.any())
            group_labels[row, slot] = bool((labels[row] & members).any())
    return group_labels, group_valid


def select_group_choices(group_scores, group_valid, representatives):
    """Choose the top valid group, breaking all ties by earliest candidate."""
    scores = np.asarray(group_scores)
    valid = np.asarray(group_valid, dtype=np.bool_)
    reps = np.asarray(representatives, dtype=np.int32)
    if scores.ndim != 2 or scores.shape != valid.shape or scores.shape != reps.shape:
        raise ValueError("group selection arrays must share shape [N, 4]")
    if not np.isfinite(scores).all():
        raise ValueError("group scores must be finite")
    choices = []
    for row in range(len(scores)):
        slots = np.flatnonzero(valid[row])
        if not len(slots):
            choices.append(0)
            continue
        slot = max(slots, key=lambda value: (float(scores[row, value]),
                                             -int(reps[row, value])))
        choices.append(int(reps[row, slot]))
    return choices

