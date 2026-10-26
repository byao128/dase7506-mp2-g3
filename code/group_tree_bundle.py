"""Fit and serialize the frozen G3 inference bundle from official train only."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from features import answer_metadata, extract_features
from group_rank_features import GROUP_RANK_FEATURE_NAMES, fit_pca
from group_tree_cv import (
    B0_CONFIG, _build_partition, _crossfit_b0, _strata,
    labels_from_representatives, registered_tree_config,
)
from group_tree_ranker import fit_ranker, save_ranker
from models import fit_scorer, save_model


def validate_bundle_inputs(primary, stability, validation, modal_config):
    """Verify that the frozen development evidence authorizes one final bundle."""
    from group_tree_final import validate_final_gates

    validate_final_gates(primary, stability)
    if validation.get("evidence_role") != "validation_development":
        raise ValueError("validation evidence role is not frozen development")
    if validation.get("test_accessed") is not False:
        raise ValueError("test must not have been accessed before bundle freeze")
    if validation.get("config") != modal_config:
        raise ValueError("validation configuration does not match the modal train config")
    if int(validation.get("g3_correct", -1)) <= int(validation.get("b0_correct", -1)):
        raise ValueError("validation did not improve over the incumbent")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_and_save_bundle(bank, embeddings, labels, config, output_dir):
    """Fit final B0/PCA/G3 assets using only labelled training rows."""
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    b0_dir = output / "b0"
    b0_dir.mkdir(parents=True, exist_ok=True)
    for marker in (output / ".gitkeep", b0_dir / ".gitkeep"):
        marker.unlink(missing_ok=True)

    metadata = answer_metadata(bank)
    candidate, candidate_names = extract_features(
        bank, embeddings, "embedding_text")
    keep = [index for index, name in enumerate(candidate_names)
            if name not in {"valid", "votes"}]
    b0_names = [candidate_names[index] for index in keep]
    b0_x = candidate[:, :, keep]
    indices = np.arange(len(bank), dtype=np.int64)
    b0_oof = _crossfit_b0(
        b0_x, b0_names, labels, metadata["valid"], indices,
        _strata(metadata), 6100, seed_offset=0)
    b0_model = fit_scorer(
        b0_x, labels, metadata["valid"],
        {**B0_CONFIG, "feature_names": b0_names})
    pca = fit_pca(embeddings)
    group_features = _build_partition(
        bank, embeddings, candidate, candidate_names, metadata,
        indices, b0_oof, pca)
    group_labels = labels_from_representatives(
        labels, group_features["group_valid"],
        group_features["representatives"])
    hard = group_features["group_valid"].sum(axis=1) >= 2
    tree = fit_ranker(
        group_features["values"][hard],
        group_features["group_valid"][hard],
        group_labels[hard],
        registered_tree_config(
            config["depth"], config["l2_leaf_reg"], config["tree_count"]),
    )

    save_model(b0_model, b0_dir)
    np.savez(
        output / "pca.npz",
        mean=np.asarray(pca["mean"], dtype=np.float32),
        std=np.asarray(pca["std"], dtype=np.float32),
        components=np.asarray(pca["components"], dtype=np.float32),
        explained_variance=np.asarray(pca["explained_variance"], dtype=np.float32),
    )
    save_ranker(tree, output / "tree.json")
    inference = {
        "format_version": 1,
        "protocol": "dase7506-mp2-open-hard-verifier-v5",
        "method": "G3 QuerySoftMax answer-group reranker",
        "feature_mode": "embedding_text",
        "exclude_features": ["valid", "votes"],
        "b0_feature_names": b0_names,
        "group_feature_count": len(GROUP_RANK_FEATURE_NAMES),
        "config": dict(config),
        "seeds": {"b0": 2028, "b0_oof_split": 6100, "tree": 2028},
        "training_rows": int(len(bank)),
        "hard_training_rows": int(hard.sum()),
        "zero_vector_groups": int(group_features["zero_vector_groups"]),
    }
    (output / "inference.json").write_text(
        json.dumps(inference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    asset_paths = [
        b0_dir / "model.json", b0_dir / "weights.npz",
        output / "pca.npz", output / "tree.json", output / "inference.json",
    ]
    manifest = {
        "format": "group-tree-v1",
        "files": {
            path.relative_to(output).as_posix(): {
                "bytes": path.stat().st_size, "sha256": _sha256(path),
            }
            for path in asset_paths
        },
        "asset_bytes_excluding_manifest": int(
            sum(path.stat().st_size for path in asset_paths)),
        "fit_seconds": float(time.perf_counter() - started),
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    manifest["asset_bytes_total"] = int(sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()))
    return manifest
