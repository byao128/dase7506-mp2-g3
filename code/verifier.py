"""Conservative exact arithmetic evidence for candidate reasoning traces."""

import ast
from fractions import Fraction
import math
import operator
import re

import numpy as np


MAX_EXPRESSION_CHARS = 256
MAX_AST_NODES = 63
MAX_INTEGER_BITS = 256
MAX_ABS_EXPONENT = 12

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:\s*%)?"
_ATOM = rf"[+\-]?(?:{_NUMBER}|\((?:[^=;\n()]|\([^=;\n()]*\)){{1,80}}\))"
_EXPRESSION = rf"{_ATOM}(?:\s*(?:\*\*|[+\-*/^×÷·⋅])\s*{_ATOM})*"
_EQUATION_CHAIN = re.compile(rf"(?P<chain>{_EXPRESSION}(?:\s*=\s*{_EXPRESSION})+)")
_VALID_COMMA_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_PERCENT = re.compile(r"(?<![\w.])([+\-]?(?:\d+(?:\.\d+)?|\.\d+))\s*%")
_DECIMAL_TOKEN = re.compile(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])")


def normalize_numeric_expression(text: str) -> str | None:
    """Normalize the supported numeric grammar without evaluating any code."""
    expression = str(text).strip()
    if not expression or len(expression) > MAX_EXPRESSION_CHARS:
        return None
    expression = (expression.replace(r"\times", "*")
                  .replace(r"\cdot", "*")
                  .replace(r"\div", "/")
                  .replace("×", "*").replace("·", "*").replace("⋅", "*")
                  .replace("÷", "/"))
    if "\\" in expression or "=" in expression:
        return None
    if re.search(r"[^0-9\s,+\-*/^().%]", expression):
        return None
    # Remove only syntactically valid thousands separators.  A leftover comma
    # would otherwise be parsed by Python as a tuple.
    expression = _VALID_COMMA_NUMBER.sub(lambda match: match.group().replace(",", ""),
                                         expression)
    if "," in expression:
        return None
    expression = _PERCENT.sub(lambda match: f"({match.group(1)}/100)", expression)
    if "%" in expression:
        return None
    expression = expression.replace("^", "**")
    expression = re.sub(r"\s+", "", expression)
    return expression or None


def _bounded(value: Fraction) -> Fraction:
    if (value.numerator.bit_length() > MAX_INTEGER_BITS or
            value.denominator.bit_length() > MAX_INTEGER_BITS):
        raise OverflowError("fraction exceeds verifier bounds")
    return value


def safe_fraction_eval(expression: str) -> Fraction | None:
    """Evaluate a bounded whitelist AST exactly, returning None when unchecked."""
    normalized = normalize_numeric_expression(expression)
    if normalized is None:
        return None
    # Preserve the original decimal token exactly.  Letting ast.parse create a
    # float first would merge adjacent decimals above binary64 precision.
    normalized = _DECIMAL_TOKEN.sub(
        lambda match: f"({int(match.group(1) + match.group(2))}/{10 ** len(match.group(2))})",
        normalized,
    )
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        return None

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = Fraction(node.value)
            return _bounded(value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _bounded(_UNARY[type(node.op)](visit(node.operand)))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base = visit(node.left)
            exponent = visit(node.right)
            if exponent.denominator != 1 or abs(exponent.numerator) > MAX_ABS_EXPONENT:
                raise ValueError("unsupported exponent")
            return _bounded(base ** exponent.numerator)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError
            return _bounded(_BINARY[type(node.op)](left, right))
        raise ValueError("unsupported syntax")

    try:
        return visit(tree)
    except (ArithmeticError, ValueError, OverflowError, RecursionError):
        return None


def _touches_unsupported_context(text, start, end):
    blocked = set("_\\{}[]")
    operators = set("+-*/^×÷·⋅=<>")
    if start:
        char = text[start - 1]
        if char.isalnum() or char in blocked or char in operators:
            return True
        before = text[:start].rstrip()
        if before and before[-1] in operators:
            return True
    if end < len(text):
        after = text[end:].lstrip()
        if not after or after.startswith("####"):
            return False
        char = after[0]
        if char == ".":
            # A full stop separates equations only at end of input or when it
            # is followed by whitespace.  A second decimal point (``1.0.0``)
            # or another adjacent token is part of the malformed chain.
            return len(after) > 1 and not after[1].isspace()
        if char in ";,:!?\n":
            return False
        if char.isalnum() or char in blocked or char in operators or char == "(":
            return True
    return False


def extract_equation_checks(text: str) -> list[tuple[Fraction, Fraction, bool]]:
    """Extract consecutive exact checks from fully supported equation chains."""
    source = str(text)
    checks = []
    for match in _EQUATION_CHAIN.finditer(source):
        if _touches_unsupported_context(source, match.start(), match.end()):
            continue
        pieces = [piece.strip() for piece in match.group("chain").split("=")]
        values = [safe_fraction_eval(piece) for piece in pieces]
        if any(value is None for value in values):
            continue
        for left, right in zip(values, values[1:]):
            checks.append((left, right, bool(left == right)))
    return checks


def arithmetic_evidence(text: str, final_answer: str | None) -> np.ndarray:
    """Return the fixed seven soft verifier features as finite float32 values."""
    checks = extract_equation_checks(text)
    if not checks:
        return np.zeros(7, dtype=np.float32)
    passed = sum(check[2] for check in checks)
    failed = len(checks) - passed
    try:
        final = Fraction(str(final_answer)) if final_answer is not None else None
    except (ValueError, ZeroDivisionError):
        final = None
    rhs = [check[1] for check in checks]
    return np.asarray([
        math.log1p(len(checks)),
        passed / len(checks),
        failed / len(checks),
        float(failed > 0),
        float(failed == 0),
        float(final is not None and rhs[-1] == final),
        float(final is not None and any(value == final for value in rhs)),
    ], dtype=np.float32)

