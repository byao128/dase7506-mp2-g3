"""Frozen offline inference for the G3 answer-group reranker."""

import json
from pathlib import Path

import numpy as np

from features import answer_metadata, extract_features
from group_rank_features import build_group_rank_features
from group_tree_ranker import (candidate_group_scores, load_ranker,
                               mixed_group_choices, predict_ranker)
from models import load_model, score_model


def _load_pca(path):
    with np.load(path, allow_pickle=False) as archive:
        return {
            "mean": archive["mean"].astype(np.float32, copy=True),
            "std": archive["std"].astype(np.float32, copy=True),
            "components": archive["components"].astype(np.float32, copy=True),
            "explained_variance": archive["explained_variance"].astype(
                np.float32, copy=True),
        }


def predict(bank, arrays, checkpoint):
    """Select one supplied candidate per question without reading references."""
    if checkpoint is None:
        raise ValueError("A G3 checkpoint directory is required")
    if not bank:
        return []
    checkpoint = Path(checkpoint)
    inference = json.loads(
        (checkpoint / "inference.json").read_text(encoding="utf-8"))
    if inference.get("protocol") != "dase7506-mp2-open-hard-verifier-v5":
        raise ValueError("Checkpoint protocol does not match the evaluator")
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
    candidate, names = extract_features(
        bank, embeddings, inference["feature_mode"])
    excluded = set(inference["exclude_features"])
    keep = [index for index, name in enumerate(names) if name not in excluded]
    selected_names = [names[index] for index in keep]
    if selected_names != inference["b0_feature_names"]:
        raise ValueError("B0 feature schema does not match the checkpoint")
    b0_model = load_model(checkpoint / "b0")
    if selected_names != b0_model.get("feature_names"):
        raise ValueError("B0 model columns do not match the checkpoint")
    b0_scores = score_model(candidate[:, :, keep], b0_model)
    metadata = answer_metadata(bank)
    group_features = build_group_rank_features(
        bank, embeddings, candidate, names, metadata, b0_scores,
        _load_pca(checkpoint / "pca.npz"))
    if group_features["values"].shape[2] != int(
            inference["group_feature_count"]):
        raise ValueError("G3 feature schema does not match the checkpoint")
    tree_scores = predict_ranker(
        load_ranker(checkpoint / "tree.json"),
        group_features["values"], group_features["group_valid"])
    groups = candidate_group_scores(
        b0_scores, metadata["valid"], metadata["group_index"])
    choices = mixed_group_choices(
        groups["scores"], tree_scores, groups["group_valid"],
        groups["representatives"], inference["config"]["alpha"])
    return [int(choice) for choice in choices]
