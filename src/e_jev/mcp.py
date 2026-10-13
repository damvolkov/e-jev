"""e_jev.mcp: the System One decision as MCP tools — noul, choice, score, models over stdio.

Claude Code, OpenCode and OpenHuman mount the same calibrated decision their SDKs see: one process, stdio
transport, every tool one System One question against `EJEV_URL` (`http://localhost:45160/v1` by default).
The answers are the endpoint's — routing, calibration and all — this module only reshapes them into tool
output. `ejev-mcp` is the console entry; the environment decides the endpoint, a bearer key included.
"""

import argparse
import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from typesafe_sdk import AsyncTypeSafeClient

from e_jev.cli.services.decide import DecideService

DEFAULT = {"url": "http://localhost:45160/v1", "key": "local", "model": "jev-latest"}


class Endpoint:
    """The one client of the process: every tool asks it the same way the CLI does."""

    __slots__ = ("_client", "_model")

    def __init__(self, client: AsyncTypeSafeClient, model: str) -> None:
        self._client = client
        self._model = model

    async def decide(self, state: Any, question: Any) -> dict[str, Any]:
        response = await self._client.system_one(state, {"q": question}, model=self._model)
        return response.model_dump(mode="json")["answers"]["q"]

    async def models(self) -> list[dict[str, Any]]:
        listed = await self._client.models.list()
        return [model.model_dump(mode="json") for model in listed.models]


@asynccontextmanager
async def lifespan(_server: MCPServer[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """The process's endpoint, opened once and closed on exit; every tool receives it through its context."""
    env = {key: os.environ.get(f"EJEV_{key.upper()}", value) for key, value in DEFAULT.items()}
    client = AsyncTypeSafeClient(api_key=env["key"], base_url=env["url"], model=env["model"], timeout=60.0)
    try:
        yield {"endpoint": Endpoint(client, env["model"])}
    finally:
        await client.aclose()


server = MCPServer(
    name="e-jev",
    instructions="Typed, calibrated decisions over a state: noul (probability true), choice (one of the options), "
    "score (a level of an ordered scale). Every answer carries the distribution; low confidence means ask a human.",
    lifespan=lifespan,
)


def _endpoint(ctx: Context[dict[str, Any]]) -> Endpoint:
    return ctx.request_context.lifespan_context["endpoint"]


@server.tool()
async def noul(ctx: Context[dict[str, Any]], state: str, question: str, true: str = "", false: str = "") -> dict[str, Any]:
    """Probability that the question is true of the state; optional criteria text for each side."""
    return await _endpoint(ctx).decide(DecideService.state(state), DecideService.noul(question, true, false))


@server.tool()
async def choice(ctx: Context[dict[str, Any]], state: str, question: str, options: list[str]) -> dict[str, Any]:
    """Pick the option that best fits the state; options as 'name' or 'name=description', 2 to 255."""
    return await _endpoint(ctx).decide(DecideService.state(state), DecideService.choice(question, tuple(options)))


@server.tool()
async def score(ctx: Context[dict[str, Any]], state: str, question: str, levels: list[str]) -> dict[str, Any]:
    """Rate the state on an ordered scale, lowest level first, 2 to 10 levels."""
    return await _endpoint(ctx).decide(DecideService.state(state), DecideService.score(question, tuple(levels)))


@server.tool()
async def models(ctx: Context[dict[str, Any]]) -> dict[str, Any]:
    """The decision models the endpoint advertises."""
    return {"models": await _endpoint(ctx).models()}


def main() -> None:
    argparse.ArgumentParser(prog="ejev-mcp", description=__doc__.splitlines()[0]).parse_args()
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
