"""logic.calibration: temperature scaling over option scores — fit by NLL, judged by ECE. Pure, no I/O."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from e_jev.models.reading import Scores

WIDTH = 26


def pad_scores(scores: Sequence[Scores]) -> Scores:
    """Ragged option scores -> one (n, 26) matrix; -inf pads carry zero mass."""
    return np.array([np.pad(score, (0, WIDTH - score.size), constant_values=-np.inf) for score in scores])


def scale_temperature(scores: Scores, temperature: float) -> Scores:
    return softmax(scores / temperature, axis=-1)


def loss_temperature(matrix: Scores, labels: NDArray[np.intp], temperature: float) -> float:
    """Mean negative log-likelihood of the labels at one temperature."""
    return float(-log_softmax(matrix / temperature, axis=-1)[np.arange(labels.size), labels].mean())


def error_calibration(probs: Scores, labels: NDArray[np.intp], bins: int = 15) -> float:
    """Expected calibration error of the top choice, equal-width confidence bins."""
    confidence, hit = probs.max(axis=-1), (probs.argmax(axis=-1) == labels).astype(np.float64)
    index = np.minimum((confidence * bins).astype(np.intp), bins - 1)
    gap = np.abs(np.bincount(index, hit, bins) - np.bincount(index, confidence, bins))
    return float(gap.sum() / labels.size)


def fit_temperature(matrix: Scores, labels: NDArray[np.intp]) -> float:
    """The temperature minimizing NLL, bounded to a sane range."""
    return float(minimize_scalar(lambda t: loss_temperature(matrix, labels, t), bounds=(0.05, 20.0), method="bounded").x)
