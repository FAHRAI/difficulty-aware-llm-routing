"""Router inputs: head+tail text and fixed hand-crafted complexity features (estimator HEUR)."""

from __future__ import annotations

import re

import numpy as np

HEAD_TAIL_LIMIT = 2000
HEAD_TAIL_PART = 1000


def head_tail(text: str) -> str:
    """Keep the question that follows a long context: first and last 1000 characters."""
    if len(text) <= HEAD_TAIL_LIMIT:
        return text
    return text[:HEAD_TAIL_PART] + " … " + text[-HEAD_TAIL_PART:]


_MATH = re.compile(r"[\\$^_=+*/<>]|\\frac|\\sqrt")
_OPTION = re.compile(r"^\s*\(?[A-J][\):.]", re.M)
_CODE = re.compile(r"```|def |class |#include|function\s*\(|;\s*$", re.M)
_QWORDS = re.compile(r"\b(why|how|explain|prove|derive|compare|analy[sz]e)\b", re.I)
_CONSTRAINTS = re.compile(r"\b(must|should|exactly|at least|at most|no more than|format|words?)\b", re.I)

HEUR_NAMES = [
    "log_chars",
    "log_words",
    "mean_word_len",
    "n_sentences",
    "n_lines",
    "digit_frac",
    "upper_frac",
    "math_symbols",
    "n_options",
    "has_code",
    "n_question_marks",
    "reasoning_words",
    "constraint_words",
    "long_context",
    "type_token_ratio",
    "max_number_len",
]


def heuristic_features(text: str) -> np.ndarray:
    words = text.split()
    n_chars, n_words = len(text), max(len(words), 1)
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return np.array(
        [
            np.log1p(n_chars),
            np.log1p(len(words)),
            sum(len(w) for w in words) / n_words,
            len(re.findall(r"[.!?](\s|$)", text)),
            text.count("\n") + 1,
            sum(c.isdigit() for c in text) / max(n_chars, 1),
            sum(c.isupper() for c in text) / max(n_chars, 1),
            len(_MATH.findall(text)),
            len(_OPTION.findall(text)),
            float(bool(_CODE.search(text))),
            text.count("?"),
            len(_QWORDS.findall(text)),
            len(_CONSTRAINTS.findall(text)),
            float(n_chars > HEAD_TAIL_LIMIT),
            len({w.lower() for w in words}) / n_words,
            max((len(x) for x in numbers), default=0),
        ],
        dtype=float,
    )


def heuristic_matrix(texts) -> np.ndarray:
    return np.vstack([heuristic_features(t) for t in texts])
