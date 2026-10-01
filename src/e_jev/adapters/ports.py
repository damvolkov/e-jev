"""adapters.ports: what the core needs from a model server — nothing about which one."""

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

from e_jev.core.settings import Settings
from e_jev.models.reading import Reading
from e_jev.models.systemone import Json


@runtime_checkable
class Reader(Protocol):
    @property
    def stop(self) -> int:
        """The token that closes an answer turn."""
        ...

    def label(self, text: str) -> tuple[int, ...]:
        """The token sequence of one option label, as the model writes it."""
        ...

    async def encode(self, prompt: str) -> tuple[int, ...]:
        """The prompt as a user turn, rendered by the chat template up to the start of the answer."""
        ...

    async def logprobs(self, tokens: Sequence[int], candidates: Sequence[int]) -> Reading:
        """Next-token log-probabilities after `tokens`, one per candidate, in candidate order."""
        ...

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json:
        """A value generated under `schema`."""
        ...

    async def health(self) -> bool: ...


type OpenReader = Callable[[Settings], AbstractAsyncContextManager[Reader]]
