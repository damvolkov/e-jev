"""models.extract: typed extraction, beyond Jev — a JSON Schema filled through Outlines."""

from typing import Any

from msgspec import Struct

from e_jev.models.systemone import Json


class ExtractRequest(Struct, frozen=True):
    state: Json
    instructions: Json
    schema: dict[str, Any]
