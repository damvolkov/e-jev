"""logic.calibration: temperature scaling over option scores — fit by NLL, judged by ECE. Pure, no I/O."""

from collections.abc import Sequence
from dataclasses import dataclass
from zlib import crc32

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from e_jev.models.reading import Scores


def pad_scores(scores: Sequence[Scores]) -> Scores:
    """Ragged option scores -> one (n, widest) matrix; -inf pads carry zero mass."""
    width = max((score.size for score in scores), default=0)
    return np.array([np.pad(score, (0, width - score.size), constant_values=-np.inf) for score in scores])


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


@dataclass(frozen=True, slots=True)
class Scored:
    """One labeled example after reading: its kind and source, raw scores, true option and split."""

    kind: str
    source: str
    scores: Scores
    label: int
    held: bool


@dataclass(frozen=True, slots=True)
class KindFit:
    temperatures: dict[str, float]
    ece_before: dict[str, float]
    ece_after: dict[str, float]
    accuracy: dict[str, float]


def split_held(key: bytes) -> bool:
    """A stable half: the same example is held out on every run, whatever the file order."""
    return crc32(key) % 2 == 1


def fit_kinds(rows: Sequence[Scored]) -> KindFit:
    """One temperature per kind fitted on the fit half; ECE before/after and accuracy per source on the held-out half."""
    kinds = sorted({row.kind for row in rows})
    part = {
        (kind, held): (pad_scores([row.scores for row in chosen]), np.array([row.label for row in chosen], dtype=np.intp))
        for kind in kinds
        for held in (False, True)
        for chosen in ([row for row in rows if row.kind == kind and row.held is held],)
    }
    temperatures = {kind: fit_temperature(*part[kind, False]) for kind in kinds}
    held = {source: [row for row in rows if row.held and row.source == source] for source in sorted({row.source for row in rows})}
    return KindFit(
        temperatures=temperatures,
        ece_before={kind: error_calibration(scale_temperature(part[kind, True][0], 1.0), part[kind, True][1]) for kind in kinds},
        ece_after={
            kind: error_calibration(scale_temperature(part[kind, True][0], temperatures[kind]), part[kind, True][1]) for kind in kinds
        },
        accuracy={
            source: float(np.mean([int(row.scores.argmax()) == row.label for row in chosen])) for source, chosen in held.items() if chosen
        },
    )
