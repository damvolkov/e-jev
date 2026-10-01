import asyncio
import importlib
import inspect
import sys
from collections.abc import Awaitable, Callable

import pytest

from e_jev.cli.main import CARDS


@pytest.mark.parametrize("card", CARDS, ids=[card.path for card in CARDS])
async def test_main_card_target_is_a_coroutine_function(card) -> None:
    module, _, function = card.target.partition(":")
    assert inspect.iscoroutinefunction(getattr(importlib.import_module(module), function))


async def test_main_build_imports_no_command_module() -> None:
    probe = "import sys; from e_jev.cli.main import build; build(); sys.exit(len([m for m in sys.modules if m.startswith('e_jev.cli.commands')]))"
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", probe)
    assert await process.wait() == 0


async def test_main_help_lists_every_command(ejev: Callable[..., Awaitable[tuple[int, str, str]]]) -> None:
    code, out, _ = await ejev("--help")
    assert (code, all(card.name in out for card in CARDS)) == (0, True)


@pytest.mark.parametrize(
    ("tokens", "code", "message"),
    [
        (("choice", "q?", "only", "--state", "s"), 2, "2 to 255"),
        (("score", "q?", "one", "--state", "s"), 2, "2 to 10"),
        (("noul", "q?", "--state", "@/no/such/file.txt"), 2, "cannot read state file"),
        (("--url", "http://127.0.0.1:9", "--timeout", "2", "models"), 3, "127.0.0.1:9"),
    ],
    ids=["one-option", "one-level", "missing-state-file", "endpoint-down"],
)
async def test_main_failures_exit_with_their_code(
    ejev: Callable[..., Awaitable[tuple[int, str, str]]], tokens: tuple[str, ...], code: int, message: str
) -> None:
    exit_code, _, err = await ejev(*tokens)
    assert (exit_code, message in err) == (code, True)
