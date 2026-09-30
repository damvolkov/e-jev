"""api.router.systemone: POST /v1/systemone and GET /v1/models — the TypeSafe surface."""

from datetime import date
from typing import Final

from litestar import get, post
from litestar.di import NamedDependency
from litestar.handlers import HTTPRouteHandler

from e_jev.api.security import bearer
from e_jev.core.settings import Settings
from e_jev.models.systemone import ModelList, ModelMetadata, SystemOneRequest, SystemOneResponse
from e_jev.operational.engine import Engine

RELEASED: Final = date(2026, 9, 30).isoformat()


async def evaluate(data: SystemOneRequest, engine: NamedDependency[Engine]) -> SystemOneResponse:
    return await engine.evaluate(data)


async def models(settings: NamedDependency[Settings]) -> ModelList:
    return ModelList(
        models=tuple(
            ModelMetadata(name=name, description=f"e-jev: System One readout over {settings.model} on vLLM.", release_date=RELEASED)
            for name in (*settings.aliases, settings.model)
        )
    )


ROUTES: Final[tuple[HTTPRouteHandler, ...]] = (
    post("/v1/systemone", name="systemone.evaluate", guards=[bearer], status_code=200)(evaluate),
    get("/v1/models", name="systemone.models", guards=[bearer])(models),
)
