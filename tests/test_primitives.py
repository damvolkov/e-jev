"""Primitive shapes against TypeSafe's own schema (typesafe-sdk) and reference confidence."""

import msgspec
import numpy as np
import pytest
from typesafe_sdk._core.response_types import SystemOneResponse as SdkResponse

from e_jev.models import Choice, Noul, Score, SystemOneRequest, SystemOneResponse, Usage
from e_jev.primitives import answer, choice_confidence, label_index, readout, score_confidence

DOCS_REQUEST = b"""{
  "state": {"subject": "Payouts", "message": "Help! My payouts have been failing for 3 days."},
  "model": "jev-latest",
  "questions": {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                  "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds", "technical": null, "sales": "Pricing"}},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}"""


def request() -> SystemOneRequest:
    return msgspec.json.decode(DOCS_REQUEST, type=SystemOneRequest)


def test_docs_request_decodes_into_tagged_questions() -> None:
    questions = request().questions
    assert [type(question) for question in questions.values()] == [Noul, Choice, Score]


def test_readout_options_carry_descriptions() -> None:
    questions = request().questions
    assert readout(questions["is_urgent"]).options == ("Yes — Explicitly time-sensitive", "No — No urgency expressed")
    assert readout(questions["department"]).options == ("billing — Payments, invoicing, refunds", "technical", "sales — Pricing")
    assert readout(questions["frustration"]).options == ("Calm", "Frustrated", "Very angry")


def test_answers_validate_against_official_sdk() -> None:
    questions = request().questions
    probs = {"is_urgent": np.array([0.95, 0.05]), "department": np.array([0.1, 0.85, 0.05]), "frustration": np.array([0.0, 0.95, 0.05])}
    response = SystemOneResponse(
        model="qwen3.6-27b",
        answers={name: answer(question, probs[name]) for name, question in questions.items()},
        usage=Usage(input_tokens=300, output_tokens=3),
    )
    parsed = SdkResponse.model_validate_json(msgspec.json.encode(response))
    assert parsed.nouls["is_urgent"].noul == pytest.approx(0.95)
    assert parsed.choices["department"].choice == "technical"
    assert parsed.scores["frustration"].score == pytest.approx(1.05)
    assert parsed.scores["frustration"].legend[2] == "Very angry"


@pytest.mark.parametrize(("probs", "expected"), [([0.5, 0.5], 0.0), ([1.0, 0.0, 0.0], 1.0), ([0.88, 0.12, 0.0], 0.82)])
def test_choice_confidence_matches_reference(probs: list[float], expected: float) -> None:
    assert choice_confidence(np.array(probs)) == pytest.approx(expected)


@pytest.mark.parametrize(("probs", "expected"), [([1 / 3] * 3, 0.0), ([0.0, 1.0, 0.0], 1.0), ([0.0, 0.95, 0.05], 0.925)])
def test_score_confidence_matches_reference(probs: list[float], expected: float) -> None:
    assert score_confidence(np.array(probs)) == pytest.approx(expected)


@pytest.mark.parametrize(("name", "label", "index"), [("is_urgent", "false", 1), ("department", "sales", 2), ("frustration", "1", 1)])
def test_label_index(name: str, label: str, index: int) -> None:
    assert label_index(request().questions[name], label) == index
