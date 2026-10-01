"""models.reading: one readout's raw scores, the prompts it encoded and its passes — interior, never on the wire."""

from dataclasses import dataclass

import numpy as np

### Spelled as ndarray[shape, dtype]: NumPy 2.5's NDArray is a TypeAliasType beartype cannot resolve; this form it checks.
type Scores = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


@dataclass(frozen=True, slots=True)
class Reading:
    scores: Scores
    prompts: tuple[tuple[int, ...], ...]
    calls: int
