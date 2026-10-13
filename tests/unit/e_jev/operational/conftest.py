"""Engine fixtures: the deterministic reading fleet, and an in-memory span sink."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from tests import kit

from e_jev.adapters.ports import Decider
from e_jev.models.calibration import Calibration, KindFit
from e_jev.models.systemone import QuestionKind
from e_jev.operational.engine import Engine


@pytest.fixture
def engine(deciders: dict[str, Decider]) -> Engine:
    return Engine(deciders, model="m", specs=[kit.reading_spec()], calibration=None)


@pytest.fixture
def engine_mirrored(reader) -> Engine:
    return Engine(kit.fleet_of(reader, permutations=2), model="m", specs=[kit.reading_spec()], calibration=None)


@pytest.fixture
def engine_cooled(reader) -> Engine:
    fitted = KindFit(
        temperatures=dict.fromkeys(QuestionKind, 1e6),
        ece_before={},
        ece_after={},
        accuracy={},
        fitted=50,
        held_out=50,
    )
    calibration = Calibration(permutations=1, backends={"reading": fitted})
    return Engine(kit.fleet_of(reader), model="m", specs=[kit.reading_spec()], calibration=calibration)


@pytest.fixture(scope="session")
def span_sink() -> InMemorySpanExporter:
    sink = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(sink))
    trace.set_tracer_provider(provider)
    return sink


@pytest.fixture
def spans(span_sink: InMemorySpanExporter) -> InMemorySpanExporter:
    span_sink.clear()
    return span_sink
