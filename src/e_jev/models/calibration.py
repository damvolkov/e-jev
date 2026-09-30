"""models.calibration: labeled examples in, a fitted temperature out — bound to the model that produced it."""

from pathlib import Path
from typing import Self

import msgspec
import structlog
from msgspec import Struct

from e_jev.models.systemone import Json, Question

log = structlog.get_logger()


class Labeled(Struct, frozen=True):
    """One calibration example: the choice key, the score level index, or "true"/"false" for a noul."""

    state: Json
    question: Question
    label: str


class Calibration(Struct, frozen=True):
    model: str
    permutations: int
    temperature: float
    ece_before: float
    ece_after: float
    samples: int

    @classmethod
    def load(cls, path: Path, model: str, permutations: int) -> Self | None:
        """The fit at `path`, applied only when made for this exact model and permutation count."""
        fitted = msgspec.json.decode(path.read_bytes(), type=cls) if path.is_file() else None
        match fitted:
            case Calibration(model=fit_model, permutations=fit_permutations) if (fit_model, fit_permutations) == (model, permutations):
                log.info("calibration_loaded", temperature=fitted.temperature, ece_after=fitted.ece_after)
                return fitted
            case Calibration():
                log.warning("calibration_stale", fitted_model=fitted.model, fitted_permutations=fitted.permutations)
                return None
            case None:
                log.warning("calibration_missing", path=str(path))
                return None
