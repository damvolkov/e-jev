"""models.calibration: labeled examples in, one fitted temperature per (backend, kind) out — bound to its fleet.

A fit is per backend because thresholds tuned on one never transfer to another: every entry names the
backend it was measured on, and the engine asks by (backend, kind). The file records the permutation
count it was fitted under, as before — the fleet fingerprint is the set of fitted backends."""

from pathlib import Path
from typing import Self

import msgspec
import structlog
from msgspec import Struct

from e_jev.models.systemone import Json, Question, QuestionKind

log = structlog.get_logger()


class Labeled(Struct, frozen=True):
    """One calibration example: the choice key, the score level index, or "true"/"false" for a noul."""

    state: Json
    question: Question
    label: str
    source: str = ""


class KindFit(Struct, frozen=True):
    """One backend: temperatures per kind, ECE and accuracy from its held-out half."""

    temperatures: dict[QuestionKind, float]
    ece_before: dict[QuestionKind, float]
    ece_after: dict[QuestionKind, float]
    accuracy: dict[str, float]
    fitted: int
    held_out: int


class Calibration(Struct, frozen=True):
    """The fleet's fits: fitted on one half of each backend's samples, measured on the other, held out."""

    permutations: int
    backends: dict[str, KindFit]

    @classmethod
    def load(cls, path: Path, permutations: int) -> Self | None:
        """The fits at `path`, applied only when fitted under this permutation count; otherwise refit."""
        stored = msgspec.json.decode(path.read_bytes(), type=cls) if path.is_file() else None
        match stored:
            case Calibration(permutations=fit_permutations) if fit_permutations == permutations:
                log.info(
                    "calibration_loaded", backends=sorted(stored.backends), ece_after={b: f.ece_after for b, f in stored.backends.items()}
                )
                return stored
            case Calibration():
                log.warning("calibration_stale", fitted_permutations=stored.permutations, permutations=permutations)
                return None
            case None:
                log.warning("calibration_missing", path=str(path))
                return None

    def temperature(self, backend: str, kind: QuestionKind) -> float:
        """The fitted temperature of one backend and kind: 1.0 when either was never fitted."""
        match self.backends.get(backend):
            case KindFit(temperatures=fitted):
                return fitted.get(kind, 1.0)
            case None:
                return 1.0
