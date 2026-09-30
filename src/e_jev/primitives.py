"""The three System One primitives as letter readouts: question -> options, distribution -> typed answer.

Confidence follows TypeSafe's reference definitions (typesafe-ai/system-one-adapter-python):
choice rescales the peak from uniform to certainty; score is 1 minus the mean distance from the mode,
relative to that of a uniform distribution.
"""

import numpy as np
from msgspec import Struct, json

from e_jev.calibration import Scores
from e_jev.models import Answer, Choice, ChoiceAnswer, Json, Noul, NoulAnswer, Question, Score, ScoreAnswer

NOUL = ("Yes", "No")
LABEL_NOUL = {"true": 0, "false": 1}


class Readout(Struct, frozen=True):
    task: str
    instructions: str
    options: tuple[str, ...]


def render(value: Json | None) -> str:
    match value:
        case str():
            return value
        case None:
            return ""
        case _:
            return json.format(json.encode(value).decode(), indent=2)


def described(name: str, description: Json | None) -> str:
    return f"{name} — {render(description)}" if description else name


def readout(question: Question) -> Readout:
    match question:
        case Noul(instructions=instructions, criteria=criteria):
            meaning = (criteria.true, criteria.false) if criteria else (None, None)
            return Readout(
                task="Is the statement or question true of the state?",
                instructions=render(instructions),
                options=tuple(described(name, text) for name, text in zip(NOUL, meaning, strict=True)),
            )
        case Choice(instructions=instructions, criteria=criteria):
            return Readout(
                task="Pick the option that best fits the state.",
                instructions=render(instructions),
                options=tuple(described(name, text) for name, text in criteria.items()),
            )
        case Score(instructions=instructions, criteria=criteria):
            return Readout(
                task="Rate the state on this ordered scale, lowest level first.",
                instructions=render(instructions),
                options=tuple(render(level) for level in criteria),
            )


def choice_confidence(probs: Scores) -> float:
    uniform = 1 / probs.size
    return float((probs.max() - uniform) / (1 - uniform))


def score_confidence(probs: Scores) -> float:
    levels = np.arange(probs.size)
    spread = np.abs(levels - (probs.size - 1) / 2).mean()
    return float(max(0.0, 1 - (probs * np.abs(levels - probs.argmax())).sum() / spread))


def answer(question: Question, probs: Scores) -> Answer:
    match question:
        case Noul():
            return NoulAnswer(noul=float(probs[0]))
        case Choice(criteria=criteria):
            return ChoiceAnswer(
                choice=tuple(criteria)[int(probs.argmax())],
                probabilities=dict(zip(criteria, probs.tolist(), strict=True)),
                confidence=choice_confidence(probs),
            )
        case Score(criteria=criteria):
            return ScoreAnswer(
                score=float((np.arange(probs.size) * probs).sum()),
                legend={str(level): text for level, text in enumerate(criteria)},
                probabilities={str(level): p for level, p in enumerate(probs.tolist())},
                confidence=score_confidence(probs),
            )


def label_index(question: Question, label: str) -> int:
    """Position of a calibration label among the question's options."""
    match question:
        case Noul():
            return LABEL_NOUL[label.lower()]
        case Choice(criteria=criteria):
            return tuple(criteria).index(label)
        case Score():
            return int(label)
