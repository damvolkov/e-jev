"""logic.primitives: the three System One primitives as letter readouts. Pure, no I/O.

Question -> lettered options; distribution over the letters -> typed answer. Confidence follows
TypeSafe's reference definitions (typesafe-ai/system-one-adapter-python): choice rescales the peak
from uniform to certainty; score is 1 minus the mean distance from the mode, relative to that of a
uniform distribution.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from msgspec import json

from e_jev.core.errors import ValidationError
from e_jev.logic.labels import LETTERS, select_labels
from e_jev.models.reading import Scores
from e_jev.models.systemone import Answer, Choice, ChoiceAnswer, Json, Noul, NoulAnswer, Question, Score, ScoreAnswer

NOUL: Final = ("Yes", "No")
ASK: Final = "{task} Reply with only the {unit} of your answer.\n\nState:\n{state}\n\nQuestion:\n{instructions}\n\nOptions:\n{options}"


class LabelIndexError(ValidationError):
    """A calibration label names no option of its question."""

    def __init__(self, label: str, options: tuple[str, ...]) -> None:
        super().__init__(f"label {label!r} is not one of {options}")
        self.label = label
        self.options = options


@dataclass(frozen=True, slots=True)
class Readout:
    task: str
    instructions: str
    options: tuple[str, ...]


def render_json(value: Json | None) -> str:
    match value:
        case str():
            return value
        case None:
            return ""
        case _:
            return json.format(json.encode(value).decode(), indent=2)


def describe_option(name: str, description: Json | None) -> str:
    return f"{name} — {render_json(description)}" if description else name


def read_question(question: Question) -> Readout:
    match question:
        case Noul(instructions=instructions, criteria=criteria):
            meaning = (criteria.true, criteria.false) if criteria else (None, None)
            return Readout(
                task="Is the statement or question true of the state?",
                instructions=render_json(instructions),
                options=tuple(describe_option(name, text) for name, text in zip(NOUL, meaning, strict=True)),
            )
        case Choice(instructions=instructions, criteria=criteria):
            return Readout(
                task="Pick the option that best fits the state.",
                instructions=render_json(instructions),
                options=tuple(describe_option(name, text) for name, text in criteria.items()),
            )
        case Score(instructions=instructions, criteria=criteria):
            return Readout(
                task="Rate the state on this ordered scale, lowest level first.",
                instructions=render_json(instructions),
                options=tuple(render_json(level) for level in criteria),
            )


def prompt_readout(state: str, readout: Readout, order: tuple[int, ...]) -> str:
    """The prompt for one option order: labels follow the order, the options are permuted."""
    labels = select_labels(len(order))
    options = "\n".join(f"{label}. {readout.options[index]}" for label, index in zip(labels, order, strict=True))
    unit = "letter" if labels[0] == LETTERS[0] else "number"
    return ASK.format(task=readout.task, unit=unit, state=state, instructions=readout.instructions, options=options)


def orders_readout(size: int, permutations: int) -> tuple[tuple[int, ...], ...]:
    """Option orders to ask: identity, then reversed — each position bias meets its mirror."""
    identity = tuple(range(size))
    return (identity, identity[::-1])[:permutations]


def confidence_choice(probs: Scores) -> float:
    uniform = 1 / probs.size
    return float((probs.max() - uniform) / (1 - uniform))


def confidence_score(probs: Scores) -> float:
    levels = np.arange(probs.size)
    spread = np.abs(levels - (probs.size - 1) / 2).mean()
    return float(max(0.0, 1 - (probs * np.abs(levels - probs.argmax())).sum() / spread))


def build_answer(question: Question, probs: Scores) -> Answer:
    match question:
        case Noul():
            return NoulAnswer(noul=float(probs[0]))
        case Choice(criteria=criteria):
            return ChoiceAnswer(
                choice=tuple(criteria)[int(probs.argmax())],
                probabilities=dict(zip(criteria, probs.tolist(), strict=True)),
                confidence=confidence_choice(probs),
            )
        case Score(criteria=criteria):
            return ScoreAnswer(
                score=float((np.arange(probs.size) * probs).sum()),
                legend={str(level): text for level, text in enumerate(criteria)},
                probabilities={str(level): p for level, p in enumerate(probs.tolist())},
                confidence=confidence_score(probs),
            )


def index_label(question: Question, label: str) -> int:
    """Position of a calibration label among the question's options."""
    keys: tuple[str, ...]
    match question:
        case Noul():
            keys, key = ("true", "false"), label.lower()
        case Choice(criteria=criteria):
            keys, key = tuple(criteria), label
        case Score(criteria=criteria):
            keys, key = tuple(map(str, range(len(criteria)))), label
    match key:
        case found if found in keys:
            return keys.index(found)
        case _:
            raise LabelIndexError(label, keys)
