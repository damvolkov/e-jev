import pytest

from e_jev.models.extract import ExtractRequest
from e_jev.models.systemone import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest
from e_jev.operational.engine import Engine


async def test_engine_evaluate_types_and_usage(engine: Engine, request_systemone: SystemOneRequest) -> None:
    response = await engine.evaluate(request_systemone)
    answers = response.answers
    assert (type(answers["is_urgent"]), type(answers["department"]), type(answers["frustration"])) == (
        NoulAnswer,
        ChoiceAnswer,
        ScoreAnswer,
    )
    assert (response.model, response.usage.output_tokens, response.usage.input_tokens > 0) == ("m", 3, True)


async def test_engine_evaluate_follows_position_bias(engine: Engine, request_systemone: SystemOneRequest) -> None:
    response = await engine.evaluate(request_systemone)
    assert isinstance(answer := response.answers["department"], ChoiceAnswer)
    assert answer.choice == "billing"


async def test_engine_mirrored_cancels_position_bias(engine_mirrored: Engine, request_systemone: SystemOneRequest) -> None:
    response = await engine_mirrored.evaluate(request_systemone)
    assert isinstance(answer := response.answers["is_urgent"], NoulAnswer)
    assert (answer.noul, response.usage.output_tokens) == (pytest.approx(0.5), 6)


async def test_engine_cooled_flattens_toward_uniform(engine_cooled: Engine, request_systemone: SystemOneRequest) -> None:
    response = await engine_cooled.evaluate(request_systemone)
    assert isinstance(answer := response.answers["department"], ChoiceAnswer)
    assert (engine_cooled.calibrated, answer.confidence) == (True, pytest.approx(0.0, abs=1e-5))


async def test_engine_extract(engine: Engine) -> None:
    assert await engine.extract(ExtractRequest(state="s", instructions="i", schema={"type": "object"})) == {"schema": ["type"]}
