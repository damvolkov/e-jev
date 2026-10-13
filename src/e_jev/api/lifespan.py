"""api.lifespan: the fleet is opened for the app's lifetime and the engine built over it."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from litestar import Litestar

from e_jev.adapters.ports import OpenFleet
from e_jev.adapters.sink import JsonlSink
from e_jev.core.settings import Settings
from e_jev.models.calibration import Calibration
from e_jev.operational.engine import Engine


@asynccontextmanager
async def lifespan(app: Litestar, open_fleet: OpenFleet) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    async with open_fleet(settings) as fleet, JsonlSink.open(settings) as sink:
        app.state.engine = Engine(
            fleet,
            model=settings.model,
            rows=settings.rows,
            specs=settings.backends,
            calibration=Calibration.load(settings.calibration, settings.permutations),
            decision_log=sink,
        )
        yield
