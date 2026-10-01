"""models.calibration: labeled examples in, one fitted temperature per question kind out — bound to its model."""

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


class Calibration(Struct, frozen=True):
    """Temperatures fitted on one half of the samples; ECE and accuracy measured on the other, held out."""

    model: str
    permutations: int
    temperatures: dict[QuestionKind, float]
    ece_before: dict[QuestionKind, float]
    ece_after: dict[QuestionKind, float]
    accuracy: dict[str, float]
    fitted: int
    held_out: int

    @classmethod
    def load(cls, path: Path, model: str, permutations: int) -> Self | None:
        """The fit at `path`, applied only when made for this exact model and permutation count."""
        fitted = msgspec.json.decode(path.read_bytes(), type=cls) if path.is_file() else None
        match fitted:
            case Calibration(model=fit_model, permutations=fit_permutations) if (fit_model, fit_permutations) == (model, permutations):
                log.info("calibration_loaded", temperatures=fitted.temperatures, ece_after=fitted.ece_after)
                return fitted
            case Calibration():
                log.warning("calibration_stale", fitted_model=fitted.model, fitted_permutations=fitted.permutations)
                return None
            case None:
                log.warning("calibration_missing", path=str(path))
                return None
