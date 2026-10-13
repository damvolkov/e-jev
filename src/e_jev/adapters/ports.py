"""adapters.ports: what the core needs from a model server — nothing about which one."""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

from e_jev.core.settings import Settings
from e_jev.models.decision import Decision
from e_jev.models.reading import Scores
from e_jev.models.systemone import Json, Question


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

    async def logprobs(self, tokens: Sequence[int], candidates: Sequence[int]) -> Scores:
        """Next-token log-probabilities after `tokens`, one per candidate, in candidate order."""
        ...

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json:
        """A value generated under `schema`."""
        ...

    async def health(self) -> bool: ...


@runtime_checkable
class Decider(Protocol):
    """Request level: a state and its whole set of questions in, raw option scores out — one forward per backend."""

    @property
    def backend(self) -> str:
        """The fleet id this decider answers as: the key its calibration is fitted under."""
        ...

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        """Scores in log space per question, over its options in order; capacity and contract failures raise."""
        ...

    async def health(self) -> bool: ...


@runtime_checkable
class Extractor(Protocol):
    """The capability only the generative teacher keeps: a value generated under a JSON schema."""

    async def extract(self, prompt: str, schema: dict[str, Any]) -> Json: ...


@runtime_checkable
class DecisionLog(Protocol):
    """Where every answered question is written for the distillation loop: state, question, raw scores, backend."""

    def record(self, state: str, question: Question, scores: Sequence[float], backend: str) -> None: ...


type OpenReader = Callable[[Settings], AbstractAsyncContextManager[Reader]]
type OpenFleet = Callable[[Settings], AbstractAsyncContextManager[Mapping[str, Decider]]]
