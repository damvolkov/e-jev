"""cli.runtime.launcher: the meta command every invocation passes through — global options, the run, the boundary.

cyclopts parses the global options here and hands the rest to the tree. `parse_args` returns the resolved
command (lazily imported by then), its bound arguments and the parameters it declared as injected; the
launcher opens the run's lifespan, injects those by type and awaits the command. `CliError` is caught once,
here, and so are the SDK's own failures: the message goes to stderr and the class decides the exit code."""

import inspect
from collections.abc import Callable, Coroutine, Iterable, Mapping
from inspect import BoundArguments
from typing import Annotated, Any

from cyclopts import App, Parameter
from rich.console import Console
from rich.markup import escape
from typesafe_sdk import TypeSafeError

from e_jev.cli.runtime.lifespan import lifespan
from e_jev.core.errors import CliError
from e_jev.core.settings import settings as st
from e_jev.models.command import CommandCard
from e_jev.models.session import Session

ENDPOINT_EXIT = 3


class Launcher:
    """Bound to the root app and the card table: knows each command's path without importing any command."""

    __slots__ = ("_app", "_paths")

    def __init__(self, app: App, cards: Iterable[CommandCard]) -> None:
        self._app = app
        self._paths = {card.target: card.path for card in cards}

    ##### PRIVATE #####

    def _run_path(self, command: Callable[..., object]) -> str:
        name = getattr(command, "__name__", "")
        return self._paths.get(f"{command.__module__}:{name}", name)

    @staticmethod
    async def _run_command(
        command: Callable[..., Coroutine[Any, Any, None]], bound: BoundArguments, injected: Mapping[str, type], session: Session
    ) -> int:
        """One run: the lifespan around the awaited command, its injected parameters resolved, the error boundary."""
        try:
            async with lifespan(session) as deps:
                await command(*bound.args, **bound.kwargs, **await deps.inject(injected))
        except CliError as error:
            Console(stderr=True).print(f"[bold red]✗[/] {escape(error.message)}")
            return error.exit_code
        except TypeSafeError as error:
            Console(stderr=True).print(f"[bold red]✗[/] {escape(session.url)}: {escape(str(error))}")
            return ENDPOINT_EXIT
        return 0

    ############################################################

    ##### PUBLIC #####

    async def run(
        self,
        *tokens: Annotated[str, Parameter(show=False, allow_leading_hyphen=True)],
        url: str = st.cli_url,
        key: str = st.cli_key,
        model: str = st.cli_model,
        seconds: Annotated[float, Parameter(name="--timeout")] = 60.0,
        json: bool = False,
        verbose: bool = False,
    ) -> int:
        """Global options, valid before any command.

        Args:
            url: System One endpoint: this e-jev, or https://api.typesafe.ai.
            key: bearer key for the endpoint.
            model: model to ask, as Jev names them.
            seconds: seconds to wait for an answer.
            json: print the raw response instead of a table.
            verbose: diagnostics on stderr at debug level.
        """
        command, bound, injected = self._app.parse_args(tokens)
        match inspect.iscoroutinefunction(command):
            case True:
                session = Session(
                    command=self._run_path(command), url=url, key=key, model=model, timeout=seconds, json=json, verbose=verbose
                )
                return await self._run_command(command, bound, injected, session)
            case _:
                ### cyclopts' own sync actions (help on a bare group): not a run — no session, no lifespan.
                command(*bound.args, **bound.kwargs)
                return 0
