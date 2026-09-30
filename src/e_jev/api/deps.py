"""api.deps: request-scoped wiring — handlers receive typed objects, never the raw app state."""

from enum import StrEnum
from typing import Final

from litestar.datastructures import State
from litestar.di import Provide

from e_jev.core.settings import Settings
from e_jev.operational.engine import Engine


class DepKey(StrEnum):
    ENGINE = "engine"
    SETTINGS = "settings"


def provide_engine(state: State) -> Engine:
    return state.engine


def provide_settings(state: State) -> Settings:
    return state.settings


DEPENDENCIES: Final[dict[str, Provide]] = {
    DepKey.ENGINE.value: Provide(provide_engine, sync_to_thread=False),
    DepKey.SETTINGS.value: Provide(provide_settings, sync_to_thread=False),
}
