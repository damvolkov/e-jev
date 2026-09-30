"""Temperature scaling over option scores: fit by NLL, judged by ECE. Pure numpy/scipy, no I/O."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

type Scores = NDArray[np.float64]


def pad(scores: Sequence[Scores]) -> Scores:
    """Ragged option scores -> one (n, 26) matrix; -inf pads carry zero mass."""
    matrix = np.full((len(scores), 26), -np.inf)
    for row, score in zip(matrix, scores, strict=True):
        row[: score.size] = score
    return matrix


def scale(scores: Scores, temperature: float) -> Scores:
    return softmax(scores / temperature, axis=-1)


def nll(matrix: Scores, labels: NDArray[np.intp], temperature: float) -> float:
    return float(-log_softmax(matrix / temperature, axis=-1)[np.arange(labels.size), labels].mean())


def ece(probs: Scores, labels: NDArray[np.intp], bins: int = 15) -> float:
    """Expected calibration error of the top choice, equal-width confidence bins."""
    confidence, hit = probs.max(axis=-1), (probs.argmax(axis=-1) == labels).astype(np.float64)
    index = np.minimum((confidence * bins).astype(np.intp), bins - 1)
    weight = np.bincount(index, minlength=bins)
    gap = np.abs(np.bincount(index, hit, bins) - np.bincount(index, confidence, bins))
    return float(gap.sum() / weight.sum())


def fit(matrix: Scores, labels: NDArray[np.intp]) -> float:
    """The temperature minimizing held-out NLL, bounded to a sane range."""
    return float(minimize_scalar(lambda t: nll(matrix, labels, t), bounds=(0.05, 20.0), method="bounded").x)
