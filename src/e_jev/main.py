"""e-jev entrypoint: the composition root — the only module that names a concretion.

Routers enter by explicit registration; the lifespan opens the reader and builds the engine.
Granian calls `create_app` per worker.
"""

from functools import partial

from litestar import Litestar
from litestar.datastructures import State

from e_jev.adapters.ports import OpenReader
from e_jev.adapters.reader.vllm import VllmReader
from e_jev.api.deps import DEPENDENCIES
from e_jev.api.handlers import HANDLERS
from e_jev.api.lifespan import lifespan
from e_jev.api.middlewares import MIDDLEWARES
from e_jev.api.router import extract, health, systemone
from e_jev.core.logger import setup_logger
from e_jev.core.settings import Settings
from e_jev.core.settings import settings as st


def create_app(settings: Settings = st, open_reader: OpenReader = VllmReader.open) -> Litestar:
    setup_logger(settings.env, settings.log_level)
    return Litestar(
        route_handlers=[*health.ROUTES, *systemone.ROUTES, *extract.ROUTES],
        lifespan=[partial(lifespan, open_reader=open_reader)],
        dependencies=DEPENDENCIES,
        middleware=list(MIDDLEWARES),
        exception_handlers=HANDLERS,
        state=State({"settings": settings}),
    )
