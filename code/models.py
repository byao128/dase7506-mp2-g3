"""Small NumPy candidate scorers, training losses, and portable checkpoints."""

import json
from pathlib import Path

import numpy as np


def _validate_targets(scores, labels, valid):
    scores = np.asarray(scores)
    labels = np.asarray(labels, dtype=np.bool_)
    valid = np.asarray(valid, dtype=np.bool_)
    if scores.ndim != 2 or labels.shape != scores.shape or valid.shape != scores.shape:
        raise ValueError("scores, labels, and valid must share shape [N, C]")
    if not np.issubdtype(scores.dtype, np.number) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite numeric values")
    if np.any(labels & ~valid):
        raise ValueError("correct labels must identify valid candidates")
    return scores, labels, valid


def _logsumexp(values):
    maximum = np.max(values)
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def pairwise_loss_and_gradient(scores, labels, valid):
    """Return the equal-question-weighted all-pairs logistic objective.

    Every valid positive is paired with every valid negative in its own
    question.  Questions without both classes are excluded from the
    denominator and contribute exactly zero gradient.
    """
    scores, labels, valid = _validate_targets(scores, labels, valid)
    work = scores.astype(np.float64, copy=False)
    if work.shape[0] == 0:
        raise ValueError("at least one question is required")
    gradient = np.zeros_like(work)
    losses = []
    informative = 0
    for row in range(work.shape[0]):
        positives = np.flatnonzero(labels[row] & valid[row])
        negatives = np.flatnonzero(~labels[row] & valid[row])
        if not len(positives) or not len(negatives):
            continue
        differences = work[row, positives, None] - work[row, negatives]
        losses.append(float(np.logaddexp(0.0, -differences).mean()))
        # d softplus(-d) / d d = -sigmoid(-d), computed stably.
        pair_gradient = -np.exp(-np.logaddexp(0.0, differences))
        pair_gradient /= differences.size
        gradient[row, positives] += pair_gradient.sum(axis=1)
        gradient[row, negatives] -= pair_gradient.sum(axis=0)
        informative += 1
    if informative == 0:
        return 0.0, gradient
    gradient /= informative
    return float(np.mean(losses)), gradient


def loss_and_gradient(scores, labels, valid, objective):
    """Return mean loss and its gradient with respect to candidate scores.

    Listwise normalization only includes parseable candidates and preserves all
    correct candidates. Binary BCE deliberately includes every candidate, so an
    unparseable candidate supplies a negative example during training.
    """
    scores, labels, valid = _validate_targets(scores, labels, valid)
    work = scores.astype(np.float64, copy=False)
    gradient = np.zeros_like(work)
    if work.shape[0] == 0:
        raise ValueError("at least one question is required")

    if objective == "listwise":
        if np.any(~valid.any(axis=1)) or np.any(~labels.any(axis=1)):
            raise ValueError("each question needs a valid and a correct candidate")
        # Four candidates per question make a masked row-wise softmax both exact
        # and much cheaper than entering Python once per training example.
        valid_work = np.where(valid, work, -np.inf)
        correct_work = np.where(labels, work, -np.inf)
        valid_max = np.max(valid_work, axis=1, keepdims=True)
        correct_max = np.max(correct_work, axis=1, keepdims=True)
        valid_exp = np.exp(valid_work - valid_max)
        correct_exp = np.exp(correct_work - correct_max)
        valid_sum = valid_exp.sum(axis=1, keepdims=True)
        correct_sum = correct_exp.sum(axis=1, keepdims=True)
        valid_logsum = valid_max[:, 0] + np.log(valid_sum[:, 0])
        correct_logsum = correct_max[:, 0] + np.log(correct_sum[:, 0])
        gradient = valid_exp / valid_sum - correct_exp / correct_sum
        return float((valid_logsum - correct_logsum).sum() / work.shape[0]), \
            gradient / work.shape[0]

    if objective == "binary":
        targets = labels.astype(np.float64)
        # softplus(s) - y*s is stable for both tails.
        loss = np.maximum(work, 0.0) - targets * work + np.log1p(np.exp(-np.abs(work)))
        sigmoid = np.empty_like(work)
        positive = work >= 0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-work[positive]))
        exp_scores = np.exp(work[~positive])
        sigmoid[~positive] = exp_scores / (1.0 + exp_scores)
        return float(loss.mean()), (sigmoid - targets) / work.size

    if objective == "pairwise":
        return pairwise_loss_and_gradient(work, labels, valid)

    raise ValueError(f"Unknown objective: {objective}")


def _model_scores(scaled, parameters, architecture):
    if architecture == "linear":
        return scaled @ parameters["weight"] + parameters["bias"][0], None
    hidden_pre = scaled @ parameters["weight1"] + parameters["bias1"]
    hidden = np.maximum(hidden_pre, 0.0)
    scores = hidden @ parameters["weight2"] + parameters["bias2"][0]
    return scores, (hidden_pre, hidden)


def _initialize(architecture, feature_count, hidden_dim, rng):
    if architecture == "linear":
        return {
            "weight": np.zeros(feature_count, dtype=np.float32),
            "bias": np.zeros(1, dtype=np.float32),
        }
    if architecture != "mlp":
        raise ValueError(f"Unknown architecture: {architecture}")
    weight1 = rng.normal(0.0, np.sqrt(2.0 / max(feature_count, 1)),
                         size=(feature_count, hidden_dim)).astype(np.float32)
    return {
        "weight1": weight1,
        "bias1": np.zeros(hidden_dim, dtype=np.float32),
        "weight2": rng.normal(0.0, np.sqrt(2.0 / max(hidden_dim, 1)),
                              size=hidden_dim).astype(np.float32),
        "bias2": np.zeros(1, dtype=np.float32),
    }


def _parameter_gradients(scaled, score_gradient, parameters, architecture, cache, l2):
    flat_x = scaled.reshape(-1, scaled.shape[-1])
    flat_gradient = score_gradient.reshape(-1).astype(np.float32)
    if architecture == "linear":
        return {
            "weight": flat_x.T @ flat_gradient + l2 * parameters["weight"],
            "bias": np.array([flat_gradient.sum()], dtype=np.float32),
        }
    hidden_pre, hidden = cache
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    hidden_gradient = flat_gradient[:, None] * parameters["weight2"][None, :]
    hidden_gradient *= hidden_pre.reshape(flat_hidden.shape) > 0
    return {
        "weight1": flat_x.T @ hidden_gradient + l2 * parameters["weight1"],
        "bias1": hidden_gradient.sum(axis=0),
        "weight2": flat_hidden.T @ flat_gradient + l2 * parameters["weight2"],
        "bias2": np.array([flat_gradient.sum()], dtype=np.float32),
    }


def fit_scorer(x, labels, valid, config, monitor=None):
    """Fit a linear or one-hidden-layer ReLU scorer with mini-batch Adam."""
    x = np.asarray(x)
    labels = np.asarray(labels, dtype=np.bool_)
    valid = np.asarray(valid, dtype=np.bool_)
    if x.ndim != 3 or labels.shape != x.shape[:2] or valid.shape != x.shape[:2]:
        raise ValueError("expected x [N, C, D] and aligned labels/valid [N, C]")
    if x.shape[0] == 0 or x.shape[2] == 0 or not np.isfinite(x).all():
        raise ValueError("training features must be nonempty and finite")

    architecture = str(config.get("architecture", "linear"))
    objective = str(config.get("objective", "listwise"))
    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 128))
    learning_rate = float(config.get("learning_rate", 0.001))
    l2 = float(config.get("l2", 0.0))
    hidden_dim = int(config.get("hidden_dim", 64))
    seed = int(config.get("seed", 0))
    monitor_training_loss = bool(config.get("monitor_training_loss", True))
    if epochs < 0 or batch_size <= 0 or learning_rate <= 0 or l2 < 0 or hidden_dim <= 0:
        raise ValueError("invalid optimization configuration")
    feature_names = list(config.get("feature_names", []))
    if feature_names and len(feature_names) != x.shape[2]:
        raise ValueError("feature_names must match the final feature dimension")

    values = x.astype(np.float32, copy=False)
    scale_mean = values.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale_std = values.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    scale_std[scale_std < np.float32(1e-6)] = np.float32(1.0)
    scaled = ((values - scale_mean) / scale_std).astype(np.float32)
    rng = np.random.default_rng(seed)
    parameters = _initialize(architecture, x.shape[2], hidden_dim, rng)
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    step = 0
    epochs_completed = 0

    for epoch in range(epochs):
        for start in range(0, x.shape[0], batch_size):
            indices = rng.permutation(x.shape[0]) if start == 0 else indices
            batch = indices[start:start + batch_size]
            scores, cache = _model_scores(scaled[batch], parameters, architecture)
            _, score_gradient = loss_and_gradient(scores, labels[batch], valid[batch], objective)
            gradients = _parameter_gradients(
                scaled[batch], score_gradient, parameters, architecture, cache, l2
            )
            step += 1
            for name in parameters:
                gradient = gradients[name].astype(np.float32, copy=False)
                first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradient
                second_moment[name] = 0.999 * second_moment[name] + 0.001 * gradient * gradient
                corrected_first = first_moment[name] / (1.0 - 0.9 ** step)
                corrected_second = second_moment[name] / (1.0 - 0.999 ** step)
                parameters[name] -= learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + np.float32(1e-8)
                )
                parameters[name] = parameters[name].astype(np.float32, copy=False)
        epochs_completed = epoch + 1
        if monitor is not None:
            snapshot = {
                "format_version": 1,
                "architecture": architecture,
                "objective": objective,
                "binary_mask_policy": "all_candidates",
                "inference_valid_policy": "mask_unparseable_before_argmax",
                "feature_names": feature_names.copy(),
                "feature_count": int(x.shape[2]),
                "hidden_dim": hidden_dim if architecture == "mlp" else 0,
                "epochs_completed": epoch + 1,
                "parameters": {name: value.copy() for name, value in parameters.items()},
                "scale_mean": scale_mean.copy(),
                "scale_std": scale_std.copy(),
            }
            info = {}
            if monitor_training_loss:
                current_scores, _ = _model_scores(scaled, parameters, architecture)
                current_loss, _ = loss_and_gradient(
                    current_scores, labels, valid, objective,
                )
                info["loss"] = current_loss
            if monitor(epoch + 1, snapshot, info) is False:
                break

    return {
        "format_version": 1,
        "architecture": architecture,
        "objective": objective,
        "binary_mask_policy": "all_candidates",
        "inference_valid_policy": "mask_unparseable_before_argmax",
        "feature_names": feature_names,
        "feature_count": int(x.shape[2]),
        "hidden_dim": hidden_dim if architecture == "mlp" else 0,
        "epochs_completed": epochs_completed,
        "parameters": parameters,
        "scale_mean": scale_mean,
        "scale_std": scale_std,
    }


def prepare_model_input(x, model):
    """Standardize a feature tensor once for repeated checkpoint scoring."""
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[2] != int(model["feature_count"]):
        raise ValueError("feature shape does not match model")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    return ((x.astype(np.float32) - model["scale_mean"]) /
            model["scale_std"]).astype(np.float32)


def score_prepared(scaled, model):
    """Score features returned by prepare_model_input for the same model scale."""
    scaled = np.asarray(scaled)
    if scaled.ndim != 3 or scaled.shape[2] != int(model["feature_count"]):
        raise ValueError("prepared feature shape does not match model")
    if not np.isfinite(scaled).all():
        raise ValueError("prepared features must be finite")
    scores, _ = _model_scores(
        scaled.astype(np.float32, copy=False), model["parameters"], model["architecture"],
    )
    return scores.astype(np.float32, copy=False)


def score_model(x, model):
    """Score [N, C, D] features with a fitted model."""
    return score_prepared(prepare_model_input(x, model), model)


def linear_model_to_raw(model):
    """Convert a standardized linear scorer to an equivalent raw-space model."""
    if model.get("architecture") != "linear":
        raise ValueError("only linear models can be collapsed")
    feature_count = int(model.get("feature_count", -1))
    names = list(model.get("feature_names", []))
    parameters = model.get("parameters", {})
    weight = np.asarray(parameters.get("weight"))
    bias = np.asarray(parameters.get("bias"))
    mean = np.asarray(model.get("scale_mean"))
    std = np.asarray(model.get("scale_std"))
    if (feature_count <= 0 or weight.shape != (feature_count,) or bias.shape != (1,) or
            mean.shape != (feature_count,) or std.shape != (feature_count,) or
            (names and len(names) != feature_count)):
        raise ValueError("linear model arrays do not match feature_count")
    arrays = (weight, bias, mean, std)
    if any(not np.issubdtype(value.dtype, np.number) or
           np.issubdtype(value.dtype, np.complexfloating) or
           not np.isfinite(value).all() for value in arrays):
        raise ValueError("linear model arrays must be finite and real")
    if np.any(std <= 0):
        raise ValueError("linear model scale_std must be positive")
    raw_weight = weight.astype(np.float64) / std.astype(np.float64)
    raw_bias = float(bias[0]) - float(np.dot(mean.astype(np.float64), raw_weight))
    if not np.isfinite(raw_weight).all() or not np.isfinite(raw_bias):
        raise ValueError("raw linear model is not finite")
    result = {
        key: value for key, value in model.items()
        if key not in {"parameters", "scale_mean", "scale_std", "provenance"}
    }
    result.update({
        "architecture": "linear",
        "feature_count": feature_count,
        "feature_names": names,
        "hidden_dim": 0,
        "parameters": {
            "weight": raw_weight.astype(np.float32),
            "bias": np.asarray([raw_bias], dtype=np.float32),
        },
        "scale_mean": np.zeros(feature_count, dtype=np.float32),
        "scale_std": np.ones(feature_count, dtype=np.float32),
    })
    return result


def average_linear_models(models, member_hashes=None):
    """Average linear scorers in raw space and retain no member parameters."""
    members = list(models)
    if not members:
        raise ValueError("at least one linear model is required")
    raw = [linear_model_to_raw(model) for model in members]
    feature_count = raw[0]["feature_count"]
    feature_names = raw[0]["feature_names"]
    for model in raw[1:]:
        if (model["feature_count"] != feature_count or
                model["feature_names"] != feature_names):
            raise ValueError("linear ensemble members must share feature columns")
    weights = np.stack([model["parameters"]["weight"].astype(np.float64)
                        for model in raw])
    biases = np.asarray([float(model["parameters"]["bias"][0]) for model in raw],
                        dtype=np.float64)
    result = {
        "format_version": max(int(model.get("format_version", 1)) for model in raw),
        "architecture": "linear",
        "objective": raw[0].get("objective", "listwise"),
        "binary_mask_policy": raw[0].get("binary_mask_policy", "all_candidates"),
        "inference_valid_policy": raw[0].get(
            "inference_valid_policy", "mask_unparseable_before_argmax"
        ),
        "feature_names": feature_names.copy(),
        "feature_count": feature_count,
        "hidden_dim": 0,
        "epochs_completed": 0,
        "parameters": {
            "weight": weights.mean(axis=0).astype(np.float32),
            "bias": np.asarray([biases.mean()], dtype=np.float32),
        },
        "scale_mean": np.zeros(feature_count, dtype=np.float32),
        "scale_std": np.ones(feature_count, dtype=np.float32),
        "provenance": {"member_count": len(raw)},
    }
    if member_hashes is not None:
        hashes = [str(value) for value in member_hashes]
        if len(hashes) != len(raw):
            raise ValueError("member_hashes must align with models")
        result["provenance"]["member_hashes"] = hashes
    return result


def save_model(model, directory):
    """Save numeric arrays separately from JSON metadata without pickle."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "scale_mean": np.asarray(model["scale_mean"], dtype=np.float32),
        "scale_std": np.asarray(model["scale_std"], dtype=np.float32),
    }
    arrays.update({f"parameter__{name}": np.asarray(value, dtype=np.float32)
                   for name, value in model["parameters"].items()})
    np.savez(directory / "weights.npz", **arrays)
    metadata = {key: value for key, value in model.items()
                if key not in {"parameters", "scale_mean", "scale_std"}}
    (directory / "model.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_model(directory):
    """Load a checkpoint whose arrays never require object deserialization."""
    directory = Path(directory)
    model = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    with np.load(directory / "weights.npz", allow_pickle=False) as archive:
        model["scale_mean"] = archive["scale_mean"].astype(np.float32, copy=True)
        model["scale_std"] = archive["scale_std"].astype(np.float32, copy=True)
        model["parameters"] = {
            name.removeprefix("parameter__"): archive[name].astype(np.float32, copy=True)
            for name in archive.files if name.startswith("parameter__")
        }
    return model
