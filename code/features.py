"""Answer grouping and deterministic candidate features."""

import ast
from collections import Counter
from fractions import Fraction
import math
import operator
import re

import numpy as np

from common import parse_answer
from verifier import arithmetic_evidence


_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_EQUATION_RE = re.compile(
    r"(?<![\w.])([+\-]?(?:\d+(?:\.\d+)?|\([^=\n]{1,80}\))"
    r"(?:\s*[+\-*/]\s*(?:\d+(?:\.\d+)?|\([^=\n]{1,80}\)))+)"
    r"\s*=\s*([+\-]?\d[\d,]*(?:\.\d+)?(?:[eE][+\-]?\d+)?(?:/\d+)?)"
    r"(?![\w,]|\.\d)"
)
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_TEXT_NAMES = [
    "valid",
    "votes",
    "text.log_length",
    "text.has_boxed",
    "text.has_hash_answer",
    "text.number_count",
    "text.answer_log_magnitude",
    "text.answer_negative",
    "text.question_number_overlap",
    "text.equation_present",
    "text.equation_consistent",
]
_VERIFIER_NAMES = [
    "verifier.log_checked_count",
    "verifier.pass_ratio",
    "verifier.fail_ratio",
    "verifier.any_failure",
    "verifier.all_checked_consistent",
    "verifier.last_checked_matches_final",
    "verifier.any_checked_matches_final",
]


def answer_metadata(bank) -> dict:
    """Return validity, answer-group support, group IDs, and disagreement flags."""
    n = len(bank)
    valid = np.zeros((n, 4), dtype=np.bool_)
    votes = np.zeros((n, 4), dtype=np.float32)
    group_index = np.full((n, 4), -1, dtype=np.int32)
    hard_mask = np.zeros(n, dtype=np.bool_)

    for row_index, row in enumerate(bank):
        candidates = row.get("candidates", [])
        if len(candidates) != 4:
            raise ValueError("Expected exactly four candidates per question")
        answers = [parse_answer(candidate.get("text", "")) for candidate in candidates]
        counts = Counter(answer for answer in answers if answer is not None)
        group_ids = {}
        for candidate_index, answer in enumerate(answers):
            if answer is None:
                continue
            valid[row_index, candidate_index] = True
            votes[row_index, candidate_index] = counts[answer]
            if answer not in group_ids:
                group_ids[answer] = len(group_ids)
            group_index[row_index, candidate_index] = group_ids[answer]
        hard_mask[row_index] = len(group_ids) >= 2

    return {
        "valid": valid,
        "votes": votes,
        "group_index": group_index,
        "hard_mask": hard_mask,
    }


def _safe_arithmetic(expression):
    """Evaluate a tiny arithmetic grammar; names, calls and exponentiation are banned."""
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError):
        return None
    if sum(1 for _ in ast.walk(tree)) > 31:
        return None

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError
            return _BINARY_OPERATORS[type(node.op)](left, right)
        raise ValueError("Unsafe arithmetic expression")

    try:
        return visit(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _equation_features(text):
    equations = _EQUATION_RE.findall(text)
    if not equations:
        return 0.0, 0.0
    checks = []
    for left, right in equations:
        left_value = _safe_arithmetic(left)
        try:
            right_value = Fraction(right.replace(",", ""))
        except (ValueError, ZeroDivisionError):
            return 1.0, 0.0
        if left_value is None:
            return 1.0, 0.0
        checks.append(left_value == right_value)
    return 1.0, float(bool(checks) and all(checks))


def _fraction(answer):
    try:
        return Fraction(answer) if answer is not None else None
    except (ValueError, ZeroDivisionError):
        return None


def _log1p_fraction_magnitude(value):
    """Compute log(1 + abs(value)) without converting huge integers to float."""
    value = abs(value)
    if value == 0:
        return 0.0
    numerator = value.numerator
    denominator = value.denominator

    def log_integer(integer):
        shift = max(integer.bit_length() - 53, 0)
        return math.log(float(integer >> shift)) + shift * math.log(2.0)

    return log_integer(numerator + denominator) - log_integer(denominator)


def _text_features(bank, metadata):
    values = np.zeros((len(bank), 4, len(_TEXT_NAMES)), dtype=np.float32)
    for row_index, row in enumerate(bank):
        question_numbers = {
            token.replace(",", "") for token in _NUMBER_RE.findall(row.get("question", ""))
        }
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = str(candidate.get("text", ""))
            answer = parse_answer(text)
            numeric = _fraction(answer)
            tokens = [token.replace(",", "") for token in _NUMBER_RE.findall(text)]
            overlap = (sum(token in question_numbers for token in tokens) / len(tokens)
                       if tokens else 0.0)
            equation_present, equation_consistent = _equation_features(text)
            values[row_index, candidate_index] = [
                float(metadata["valid"][row_index, candidate_index]),
                metadata["votes"][row_index, candidate_index],
                math.log1p(len(text)),
                float("\\boxed{" in text),
                float("####" in text),
                math.log1p(len(tokens)),
                _log1p_fraction_magnitude(numeric) if numeric is not None else 0.0,
                float(numeric is not None and numeric < 0),
                overlap,
                equation_present,
                equation_consistent,
            ]
    return values


def _verifier_features(bank):
    values = np.zeros((len(bank), 4, len(_VERIFIER_NAMES)), dtype=np.float32)
    for row_index, row in enumerate(bank):
        for candidate_index, candidate in enumerate(row["candidates"]):
            text = str(candidate.get("text", ""))
            values[row_index, candidate_index] = arithmetic_evidence(text, parse_answer(text))
    return values


def extract_features(bank, embeddings, mode: str):
    """Build candidate features without fitting any data-dependent statistics."""
    if mode not in {"embedding", "embedding_text", "relative_text",
                    "embedding_text_verifier", "relative_text_verifier"}:
        raise ValueError(f"Unknown feature mode: {mode}")
    embeddings = np.asarray(embeddings)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (len(bank), 4):
        raise ValueError("Expected embeddings with shape [N, 4, D]")
    if (not np.issubdtype(embeddings.dtype, np.number) or
            np.issubdtype(embeddings.dtype, np.complexfloating)):
        raise ValueError("Embeddings must be real numeric values")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be finite")
    embeddings = embeddings.astype(np.float32, copy=False)
    embedding_names = [f"embedding.{index}" for index in range(embeddings.shape[2])]
    if mode == "embedding":
        return embeddings.copy(), embedding_names

    metadata = answer_metadata(bank)
    text = _text_features(bank, metadata)
    if mode in {"embedding_text_verifier", "relative_text_verifier"}:
        verifier = _verifier_features(bank)
        own = np.concatenate([embeddings, text, verifier], axis=2).astype(np.float32, copy=False)
        own_names = embedding_names + _TEXT_NAMES + _VERIFIER_NAMES
    else:
        own = np.concatenate([embeddings, text], axis=2).astype(np.float32, copy=False)
        own_names = embedding_names + _TEXT_NAMES
    if mode in {"embedding_text", "embedding_text_verifier"}:
        return own, own_names

    question_delta = own - own.mean(axis=1, keepdims=True)
    group_delta = np.zeros_like(own)
    for row_index in range(len(bank)):
        groups = metadata["group_index"][row_index]
        for candidate_index, group in enumerate(groups):
            if group >= 0:
                group_delta[row_index, candidate_index] = (
                    own[row_index, candidate_index] - own[row_index, groups == group].mean(axis=0)
                )
    result = np.concatenate([own, question_delta, group_delta], axis=2)
    names = ([f"self.{name}" for name in own_names] +
             [f"question_delta.{name}" for name in own_names] +
             [f"group_delta.{name}" for name in own_names])
    return result.astype(np.float32, copy=False), names
