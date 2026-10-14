"""models.calibration: labeled examples in, one fitted temperature per (backend, kind) out — bound to its fleet.

A fit is per backend because thresholds tuned on one never transfer to another — and it names the model it
was fitted ON, because a route pointed at a different model must not inherit its temperatures. The engine
applies the whole file only when the fleet's id->model map matches what was fitted; otherwise it serves
uncalibrated and says so."""

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
    """One backend: the model it was fitted on, temperatures per kind, ECE and accuracy from its held-out half."""

    model: str
    temperatures: dict[QuestionKind, float]
    ece_before: dict[QuestionKind, float]
    ece_after: dict[QuestionKind, float]
    accuracy: dict[str, float]
    fitted: int
    held_out: int


class Calibration(Struct, frozen=True):
    """The fleet's fits: each bound to its backend's model, measured on the half never fitted on."""

    permutations: int
    backends: dict[str, KindFit]

    @classmethod
    def load(cls, path: Path, permutations: int) -> Self | None:
        """The fits at `path` when fitted under this permutation count; per-backend model binding is checked by the caller."""
        stored = msgspec.json.decode(path.read_bytes(), type=cls) if path.is_file() else None
        match stored:
            case Calibration(permutations=fit_permutations) if fit_permutations == permutations:
                log.info("calibration_loaded", fitted={b: f.model for b, f in stored.backends.items()})
                return stored
            case Calibration():
                log.warning("calibration_stale", fitted_permutations=stored.permutations, permutations=permutations)
                return None
            case None:
                log.warning("calibration_missing", path=str(path))
                return None

    def temperature(self, backend: str, model: str, kind: QuestionKind) -> float:
        """The fitted temperature of one backend and kind under its own model: 1.0 when anything was never fitted."""
        match self.backends.get(backend):
            case KindFit(model=fitted_model, temperatures=fitted) if fitted_model == model:
                return fitted.get(kind, 1.0)
            case KindFit(model=fitted_model):
                log.warning("calibration_model_mismatch", backend=backend, fitted=fitted_model, served=model)
                return 1.0
            case None:
                return 1.0
