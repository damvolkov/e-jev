"""cli.runtime.deps: injection by declared type — built on demand, once per run, closed by the run's stack.

`Inject[T]` marks a parameter cyclopts must not parse; the launcher resolves each from its type through
`Deps.PROVIDERS`. The TypeSafe client opens only for a command that asks for it, on the run's exit stack."""

from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from types import MappingProxyType
from typing import Annotated, ClassVar, TypeVar, cast

from cyclopts import Parameter
from rich.console import Console
from typesafe_sdk import AsyncTypeSafeClient

from e_jev.cli.services.decide import DecideService
from e_jev.core.settings import Settings
from e_jev.core.settings import settings as st
from e_jev.models.session import Session

T = TypeVar("T")
### cyclopts reads `Annotated` metadata but does not unwrap PEP 695 `type` aliases: the classic generic alias is what it sees.
Inject = Annotated[T, Parameter(parse=False)]

type Builder = Callable[[Deps], Awaitable[object]]


class Deps:
    """The run's provider: builds what a command declares, caches it for the run, owns nothing past the stack."""

    __slots__ = ("_built", "_session", "_stack")

    def __init__(self, stack: AsyncExitStack, session: Session) -> None:
        self._stack = stack
        self._session = session
        self._built: dict[Builder, object] = {}

    ##### PRIVATE #####

    async def _resolve_session(self) -> Session:
        return self._session

    async def _resolve_settings(self) -> Settings:
        return st

    async def _resolve_console(self) -> Console:
        return Console()

    async def _resolve_client(self) -> AsyncTypeSafeClient:
        client = AsyncTypeSafeClient(
            api_key=self._session.key, base_url=self._session.url, model=self._session.model, timeout=self._session.timeout
        )
        return await self._stack.enter_async_context(client)

    async def _resolve_decide(self) -> DecideService:
        return DecideService(await self.resolve(AsyncTypeSafeClient), self._session.model)

    ############################################################

    ##### PUBLIC #####

    PROVIDERS: ClassVar[Mapping[type, Builder]] = MappingProxyType(
        {
            Session: _resolve_session,
            Settings: _resolve_settings,
            Console: _resolve_console,
            AsyncTypeSafeClient: _resolve_client,
            DecideService: _resolve_decide,
        }
    )

    async def resolve[K](self, kind: type[K]) -> K:
        """The run's single instance of `kind`, built on first ask."""
        try:
            build = self.PROVIDERS[kind]
        except KeyError:
            msg = f"no provider for {kind.__name__}: add a builder to Deps.PROVIDERS"
            raise TypeError(msg) from None
        self._built[build] = self._built[build] if build in self._built else await build(self)
        return cast("K", self._built[build])

    async def inject(self, wanted: Mapping[str, type]) -> dict[str, object]:
        """Every non-parsed parameter of a command, by name, resolved from its declared type."""
        return {name: await self.resolve(kind) for name, kind in wanted.items()}
