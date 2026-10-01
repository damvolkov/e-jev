"""Engine fixtures over the fake reader, and an in-memory span sink."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from e_jev.adapters.ports import Reader
from e_jev.models.calibration import Calibration
from e_jev.models.systemone import QuestionKind
from e_jev.operational.engine import Engine


@pytest.fixture
def engine(reader: Reader) -> Engine:
    return Engine(reader, model="m", permutations=1, concurrency=4, calibration=None)


@pytest.fixture
def engine_mirrored(reader: Reader) -> Engine:
    return Engine(reader, model="m", permutations=2, concurrency=4, calibration=None)


@pytest.fixture
def engine_cooled(reader: Reader) -> Engine:
    fitted = Calibration(
        model="m",
        permutations=1,
        temperatures=dict.fromkeys(QuestionKind, 1e6),
        ece_before={},
        ece_after={},
        accuracy={},
        fitted=50,
        held_out=50,
    )
    return Engine(reader, model="m", permutations=1, concurrency=4, calibration=fitted)


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
