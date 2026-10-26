"""Leakage-safe fixed features for answer-group ranking."""

import math

import numpy as np

from group_fusion import build_group_labels


PCA_DIM = 64
FEATURE_COUNT = 1438


def fit_pca(embeddings, n_components=PCA_DIM):
    """Fit standardization and PCA using only the supplied partition."""
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 4 or values.shape[2] != 1024:
        raise ValueError("embeddings must have shape [N, 4, 1024]")
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("PCA fit data must be nonempty and finite")
    if int(n_components) != PCA_DIM:
        raise ValueError("the registered feature schema fixes PCA at 64 columns")
    flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    scaled = (flat - mean) / std
    covariance = scaled.T @ scaled / max(len(scaled) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:PCA_DIM]
    components = eigenvectors[:, order].T
    return {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_variance": eigenvalues[order].astype(np.float32),
    }


def transform_pca(embeddings, pca):
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (4, 1024):
        raise ValueError("embeddings must have shape [N, 4, 1024]")
    mean = np.asarray(pca["mean"], dtype=np.float32)
    std = np.asarray(pca["std"], dtype=np.float32)
    components = np.asarray(pca["components"], dtype=np.float32)
    if mean.shape != (1024,) or std.shape != (1024,) or components.shape != (64, 1024):
        raise ValueError("PCA state violates the fixed 64x1024 schema")
    return (((values - mean) / std) @ components.T).astype(np.float32)


def _logsumexp(values):
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values.astype(np.float64) - maximum).sum()))


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _cosine(left, right):
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _pairwise_mean(vectors):
    if len(vectors) < 2:
        return 0.0
    values = [_cosine(vectors[i], vectors[j])
              for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return float(np.mean(values))


def _feature_names():
    names = [f"raw_mean.{i}" for i in range(1024)]
    for statistic in ("mean", "std", "min", "max"):
        names.extend(f"pca_{statistic}.{i}" for i in range(64))
    names.extend(f"pca_group_minus_question.{i}" for i in range(64))
    names.extend(f"pca_question_mean.{i}" for i in range(64))
    for statistic in ("mean", "max"):
        names.extend(f"text_{statistic}.{i}" for i in range(9))
    names.extend([
        "support.log_group_size", "support.parseable_share",
        "support.valid_group_count", "support.valid_candidate_count",
        "support.vote_margin", "support.singleton",
        "cosine.within_mean", "cosine.cross_mean", "cosine.cross_max",
        "b0.logsumexp", "b0.probability", "b0.margin",
    ])
    if len(names) != FEATURE_COUNT:
        raise RuntimeError("group feature schema does not total 1438")
    return names


GROUP_RANK_FEATURE_NAMES = _feature_names()


def build_unique_group_labels(candidate_labels, metadata):
    group_labels, group_valid = build_group_labels(candidate_labels, metadata)
    for row in range(len(group_labels)):
        if group_valid[row].sum() >= 1 and group_labels[row, group_valid[row]].sum() != 1:
            raise ValueError("each supervised question must have exactly one correct group")
    return group_labels


def build_group_rank_features(bank, embeddings, candidate_features, candidate_names,
                              metadata, b0_candidate_scores, pca):
    """Return the registered 1438-column tensor in deterministic group slots."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    candidate = np.asarray(candidate_features, dtype=np.float32)
    scores = np.asarray(b0_candidate_scores, dtype=np.float32)
    n = len(bank)
    if embeddings.shape != (n, 4, 1024) or candidate.shape[:2] != (n, 4):
        raise ValueError("candidate arrays are not aligned")
    if scores.shape != (n, 4) or not np.isfinite(scores).all():
        raise ValueError("B0 scores must be finite [N, 4]")
    names = list(candidate_names)
    text_columns = [
        i for i, name in enumerate(names)
        if not name.startswith("embedding.") and name not in {"valid", "votes"}
    ]
    if len(text_columns) != 9:
        raise ValueError("expected the fixed nine non-valid/votes text columns")
    valid = np.asarray(metadata["valid"], dtype=bool)
    groups = np.asarray(metadata["group_index"], dtype=np.int32)
    if valid.shape != (n, 4) or groups.shape != (n, 4):
        raise ValueError("metadata is not aligned")
    projected = transform_pca(embeddings, pca)
    values = np.zeros((n, 4, FEATURE_COUNT), dtype=np.float32)
    group_valid = np.zeros((n, 4), dtype=bool)
    representatives = np.full((n, 4), -1, dtype=np.int32)
    zero_vectors = 0
    for row in range(n):
        valid_members = np.flatnonzero(valid[row])
        slots = sorted(set(groups[row, valid_members].tolist()))
        question_mean = (projected[row, valid_members].mean(axis=0)
                         if len(valid_members) else np.zeros(64, dtype=np.float32))
        group_sizes = {slot: int(np.sum(valid[row] & (groups[row] == slot)))
                       for slot in slots}
        group_base = {}
        for slot in slots:
            members = np.flatnonzero(valid[row] & (groups[row] == slot))
            group_base[slot] = _logsumexp(scores[row, members])
        probabilities = (_softmax([group_base[slot] for slot in slots])
                         if slots else np.asarray([]))
        for position, slot in enumerate(slots):
            members = np.flatnonzero(valid[row] & (groups[row] == slot))
            other = np.asarray([i for i in valid_members if i not in set(members)], dtype=int)
            group_valid[row, slot] = True
            representatives[row, slot] = int(members[0])
            raw = embeddings[row, members]
            low = projected[row, members]
            text = candidate[row, members][:, text_columns]
            cross = [_cosine(embeddings[row, left], embeddings[row, right])
                     for left in members for right in other]
            zero_vectors += int(np.any(np.linalg.norm(raw, axis=1) <= 1e-12))
            other_sizes = [size for key, size in group_sizes.items() if key != slot]
            other_base = [value for key, value in group_base.items() if key != slot]
            block = np.concatenate([
                raw.mean(axis=0),
                low.mean(axis=0), low.std(axis=0), low.min(axis=0), low.max(axis=0),
                low.mean(axis=0) - question_mean, question_mean,
                text.mean(axis=0), text.max(axis=0),
                np.asarray([
                    math.log(float(len(members))),
                    len(members) / float(max(len(valid_members), 1)),
                    len(slots), len(valid_members),
                    len(members) - (max(other_sizes) if other_sizes else 0),
                    float(len(members) == 1),
                    _pairwise_mean(raw),
                    float(np.mean(cross)) if cross else 0.0,
                    float(np.max(cross)) if cross else 0.0,
                    group_base[slot], probabilities[position],
                    group_base[slot] - (max(other_base) if other_base else group_base[slot]),
                ], dtype=np.float32),
            ]).astype(np.float32)
            if block.shape != (FEATURE_COUNT,):
                raise RuntimeError("group feature block has wrong width")
            values[row, slot] = block
    if not np.isfinite(values).all():
        raise ValueError("group features must be finite")
    return {
        "values": values,
        "feature_names": GROUP_RANK_FEATURE_NAMES.copy(),
        "group_valid": group_valid,
        "representatives": representatives,
        "zero_vector_groups": int(zero_vectors),
    }
