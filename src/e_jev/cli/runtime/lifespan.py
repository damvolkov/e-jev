"""cli.runtime.lifespan: one run's scope — logs on stderr, traces when configured, every edge closed.

A cli process lives for one command, so the lifespan is one `async with`. stdout belongs to the command's
output (tables or JSON), so logging goes to stderr. With JEV_OTLP_ENDPOINT set, the run is traced: the
official OpenInference instrumentor spans each System One call, and the request carries the context, so
e-jev's and vLLM's spans nest under it in Phoenix."""

import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

import structlog
from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from opentelemetry import trace

from e_jev.cli.runtime.deps import Deps
from e_jev.core.logger import setup_logger
from e_jev.core.settings import settings as st
from e_jev.core.telemetry import setup_telemetry
from e_jev.models.session import Session

log = structlog.get_logger()
tracer = trace.get_tracer("e_jev.cli")


@asynccontextmanager
async def lifespan(session: Session) -> AsyncIterator[Deps]:
    setup_logger("dev", "DEBUG" if session.verbose else "WARNING", stream=sys.stderr)
    async with AsyncExitStack() as stack:
        match setup_telemetry(st.otlp_endpoint, st.otlp_project, service="ejev"):
            case None:
                pass
            case provider:
                TypeSafeAIInstrumentor().instrument(tracer_provider=provider)
                stack.callback(provider.shutdown)
        with tracer.start_as_current_span(f"ejev {session.command}"):
            log.debug("run_start", command=session.command, url=session.url)
            yield Deps(stack, session)
