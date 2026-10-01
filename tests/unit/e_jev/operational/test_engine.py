import msgspec
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from e_jev.models.extract import ExtractRequest
from e_jev.models.systemone import Choice, ChoiceAnswer, Noul, NoulAnswer, Question, ScoreAnswer, SystemOneRequest
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


@pytest.mark.parametrize("size", [27, 40, 255])
async def test_engine_evaluate_choice_beyond_letters(engine: Engine, size: int) -> None:
    criteria = {f"area_{index}": None for index in range(size)}
    request = SystemOneRequest(state="s", questions={"area": Choice(criteria=criteria, instructions="Which area?")})
    response = await engine.evaluate(request)
    assert isinstance(answer := response.answers["area"], ChoiceAnswer)
    assert (answer.choice, sum(answer.probabilities.values()), len(answer.probabilities)) == ("area_0", pytest.approx(1.0), size)


async def test_engine_evaluate_traces_openinference_spans(
    engine: Engine, request_systemone: SystemOneRequest, spans: InMemorySpanExporter
) -> None:
    await engine.evaluate(request_systemone)
    finished = {span.name: span for span in spans.get_finished_spans()}
    root = finished["systemone"]
    assert (root.attributes or {})["openinference.span.kind"] == "CHAIN"
    assert isinstance(output := (root.attributes or {})["output.value"], str)
    assert set(msgspec.json.decode(output)["answers"]) == {"is_urgent", "department", "frustration"}
    assert {"question is_urgent", "question department", "question frustration", "readout"} <= set(finished)
    assert (finished["readout"].attributes or {})["openinference.span.kind"] == "LLM"


async def test_engine_evaluate_bills_the_state_once(engine: Engine) -> None:
    state = " ".join(f"fact{index}" for index in range(500))
    questions: dict[str, Question] = {f"q{index}": Noul(instructions=f"Is fact{index} true?") for index in range(10)}
    response = await engine.evaluate(SystemOneRequest(state=state, questions=questions))
    assert 500 < response.usage.input_tokens < 2 * 500
