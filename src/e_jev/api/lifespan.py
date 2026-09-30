"""api.lifespan: the reader is opened for the app's lifetime and the engine built over it."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from litestar import Litestar

from e_jev.adapters.ports import OpenReader
from e_jev.core.settings import Settings
from e_jev.models.calibration import Calibration
from e_jev.operational.engine import Engine


@asynccontextmanager
async def lifespan(app: Litestar, open_reader: OpenReader) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    async with open_reader(settings) as reader:
        app.state.engine = Engine(
            reader=reader,
            model=settings.model,
            permutations=settings.permutations,
            concurrency=settings.concurrency,
            calibration=Calibration.load(settings.calibration, settings.model, settings.permutations),
        )
        yield
