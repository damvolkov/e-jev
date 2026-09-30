"""api.router.health: liveness answers always; readiness only while the model server answers."""

from typing import Final

from litestar import Response, get
from litestar.di import NamedDependency
from litestar.handlers import HTTPRouteHandler
from litestar.status_codes import HTTP_200_OK, HTTP_503_SERVICE_UNAVAILABLE

from e_jev.operational.engine import Engine


async def health() -> dict[str, str]:
    return {"status": "ok"}


async def ready(engine: NamedDependency[Engine]) -> Response[dict[str, str]]:
    up = await engine.health()
    return Response({"status": "ready" if up else "reader_down"}, status_code=HTTP_200_OK if up else HTTP_503_SERVICE_UNAVAILABLE)


ROUTES: Final[tuple[HTTPRouteHandler, ...]] = (
    get("/health", name="health")(health),
    get("/ready", name="ready")(ready),
)
