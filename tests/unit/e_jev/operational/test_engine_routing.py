"""operational.engine over a mixed fleet: what routes to what, who falls back to whom, capacity's voice."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from tests import kit

from e_jev.config.backends import BackendKind, Language
from e_jev.core.errors import OverloadedError, RateLimitedError, RequestError
from e_jev.logic.dispatch import option_count
from e_jev.models.decision import Decision
from e_jev.models.systemone import Choice, Noul, Question, QuestionKind, SystemOneRequest
from e_jev.operational.engine import Engine, EngineEvaluateError

FAST = kit.backend("fast", BackendKind.SYSTEMONE, languages=(Language.EN,))
ES = kit.backend("es", BackendKind.LAYA, languages=(Language.ES,), max_options=20)
TEACHER = kit.backend("reading", BackendKind.READING, languages=(Language.ANY,))
SPEC_BY_ID = {spec.id: spec for spec in (FAST, ES, TEACHER)}
STATE_EN = "charged twice, refund the duplicate or we cancel"
STATE_ES = "Me han cobrado dos veces, devuelvan el duplicado"


class CountingDecider(kit.FakeDecider):
    """A routing double with fixed vectors per shape: records every subset it was asked."""

    def __init__(self, backend: str, *, shapes: Mapping[str, int] | None = None, **kwargs: Any) -> None:
        super().__init__(backend, **kwargs)
        self._shapes = shapes or {}

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        self.seen.append(tuple(questions))
        match self._failing:
            case None:
                scores = {
                    name: np.log(np.linspace(0.9, 0.1, self._shapes.get(name, option_count(question))))
                    for name, question in questions.items()
                }
                return Decision(scores=scores, input_tokens=5, output_tokens=0)
            case error:
                message = "refused"
                raise error(message)


def _engine(deciders: Mapping[str, kit.FakeDecider], rows: dict[QuestionKind, tuple[str, ...]]) -> Engine:
    return Engine(deciders, model="m", rows=rows, specs=list(SPEC_BY_ID.values()), calibration=None)


async def test_engine_routes_a_wide_choice_past_laya() -> None:
    fast, es, teacher = CountingDecider("fast"), CountingDecider("es"), CountingDecider("reading")
    engine = _engine({"fast": fast, "es": es, "reading": teacher}, {QuestionKind.CHOICE: ("es", "fast", "reading")})
    request = SystemOneRequest(
        state=STATE_ES,
        questions={
            "wide": Choice(criteria=dict.fromkeys(f"o{index}" for index in range(40))),
            "small": Choice(criteria={"a": None, "b": None}),
        },
    )
    await engine.evaluate(request)
    assert (es.seen, fast.seen, teacher.seen) == ([("small",)], [], [("wide",)])


async def test_engine_falls_back_down_the_row_when_the_head_fails() -> None:
    fast, teacher = CountingDecider("fast", failing=_refused), CountingDecider("reading")
    engine = _engine({"fast": fast, "reading": teacher}, {QuestionKind.NOUL: ("fast", "reading")})
    response = await engine.evaluate(SystemOneRequest(state=STATE_EN, questions={"q": Noul(instructions="?")}))
    assert (fast.seen, teacher.seen, "q" in response.answers) == ([("q",)], [("q",)], True)


async def test_engine_asks_the_row_head_alone_when_it_answers() -> None:
    fast, teacher = CountingDecider("fast"), CountingDecider("reading")
    engine = _engine({"fast": fast, "reading": teacher}, {QuestionKind.NOUL: ("fast", "reading")})
    await engine.evaluate(SystemOneRequest(state=STATE_EN, questions={"q": Noul(instructions="?")}))
    assert (fast.seen, teacher.seen) == ([("q",)], [])


async def test_engine_answers_capacity_when_the_whole_row_is_at_capacity() -> None:
    engine = _engine(
        {"fast": CountingDecider("fast", failing=_overload), "reading": CountingDecider("reading", failing=_ratelimit)},
        {QuestionKind.NOUL: ("fast", "reading")},
    )
    with pytest.raises(RateLimitedError) as caught:
        await engine.evaluate(SystemOneRequest(state=STATE_EN, questions={"q": Noul(instructions="?")}))
    assert caught.value.retry_after == 3.0


async def test_engine_evaluate_error_names_every_leaf() -> None:
    engine = _engine(
        {"fast": CountingDecider("fast", failing=_refused), "reading": CountingDecider("reading", failing=_refused)},
        {QuestionKind.NOUL: ("fast", "reading")},
    )
    with pytest.raises(EngineEvaluateError) as caught:
        await engine.evaluate(SystemOneRequest(state=STATE_EN, questions={"q": Noul(instructions="?")}))
    assert "refused" in caught.value.message


def _refused(message: str) -> Exception:
    return RequestError(message)


def _overload(message: str) -> Exception:
    return OverloadedError(message)


def _ratelimit(message: str) -> Exception:
    return RateLimitedError(message, retry_after=3.0)
