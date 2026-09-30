"""Litestar app: composition root. TypeSafe-compatible System One API over one engine per process."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date

import structlog
from litestar import Litestar, Request, Response, get, post
from litestar.connection import ASGIConnection
from litestar.datastructures import State
from litestar.exceptions import NotAuthorizedException, ValidationException
from litestar.handlers import BaseRouteHandler
from litestar.status_codes import HTTP_422_UNPROCESSABLE_ENTITY

from e_jev.engine import Engine
from e_jev.models import ExtractRequest, Json, ModelList, ModelMetadata, SystemOneRequest, SystemOneResponse
from e_jev.settings import Settings

RELEASED = date(2026, 9, 30).isoformat()


@asynccontextmanager
async def engine(app: Litestar) -> AsyncIterator[None]:
    app.state.engine = await Engine.connect(app.state.settings)
    try:
        yield
    finally:
        await app.state.engine.aclose()


def bearer(connection: ASGIConnection, _: BaseRouteHandler) -> None:
    match connection.app.state.settings.api_key:
        case None:
            return
        case expected if connection.headers.get("authorization") == f"Bearer {expected}":
            return
        case _:
            raise NotAuthorizedException("Missing or invalid API key.")


def unprocessable(_: Request, error: ValidationException) -> Response:
    """Validation failures as TypeSafe reports them: 422 with a `detail` list."""
    return Response(
        {"detail": [{"loc": ["body"], "msg": error.detail, "type": "value_error", "ctx": error.extra}]},
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
    )


@get("/health", sync_to_thread=False)
def health() -> dict[str, str]:
    return {"status": "ok"}


@post("/v1/systemone", guards=[bearer], status_code=200)
async def system_one(data: SystemOneRequest, state: State) -> SystemOneResponse:
    return await state.engine.system_one(data)


@get("/v1/models", guards=[bearer], sync_to_thread=False)
def models(state: State) -> ModelList:
    settings: Settings = state.settings
    return ModelList(
        models=tuple(
            ModelMetadata(name=name, description=f"e-jev: System One readout over {settings.model} on vLLM.", release_date=RELEASED)
            for name in (*settings.aliases, settings.model)
        )
    )


@post("/v1/extract", guards=[bearer], status_code=200)
async def extract(data: ExtractRequest, state: State) -> Json:
    return await state.engine.extract(data)


def create_app() -> Litestar:
    settings = Settings()
    structlog.configure(
        processors=[structlog.processors.add_log_level, structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()],
        wrapper_class=structlog.make_filtering_bound_logger(settings.log_level),
    )
    return Litestar(
        route_handlers=[health, system_one, models, extract],
        lifespan=[engine],
        state=State({"settings": settings}),
        exception_handlers={ValidationException: unprocessable},
    )
