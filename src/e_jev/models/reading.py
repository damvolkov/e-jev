"""models.reading: one readout's raw scores plus the tokens it cost — interior, never on the wire."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Scores = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Reading:
    scores: Scores
    input_tokens: int
    calls: int
