"""core.telemetry: OpenTelemetry configured once at the composition root — spans out over OTLP/HTTP.

Outbound httpx calls carry the trace context, so vLLM's own spans nest under the readout that caused them.
"""

from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_telemetry(endpoint: str | None, project: str) -> TracerProvider | None:
    match endpoint:
        case None:
            return None
        case url:
            provider = TracerProvider(resource=Resource.create({SERVICE_NAME: project, ResourceAttributes.PROJECT_NAME: project}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url)))
            trace.set_tracer_provider(provider)
            HTTPXClientInstrumentor().instrument()
            return provider
