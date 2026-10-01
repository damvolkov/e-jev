"""cli.commands.decide: driving adapter — parse the terminal, call `DecideService`, render. No rule lives here."""

from pathlib import Path

from rich.console import Console

from e_jev.cli.runtime.deps import Inject
from e_jev.cli.services.decide import DecideService
from e_jev.cli.views import render_models, render_response
from e_jev.models.session import Session


async def noul(
    question: str,
    *,
    state: str,
    true: str = "",
    false: str = "",
    decide: Inject[DecideService],
    session: Inject[Session],
    console: Inject[Console],
) -> None:
    """A yes/no question: the probability the answer is yes.

    Args:
        question: the question or statement to judge.
        state: the content, or @file (.json is structured).
        true: what a yes means.
        false: what a no means.
    """
    response = await decide.ask(decide.state(state), {"answer": decide.noul(question, true, false)})
    render_response(console, response, raw=session.json)


async def choice(
    question: str,
    *options: str,
    state: str,
    decide: Inject[DecideService],
    session: Inject[Session],
    console: Inject[Console],
) -> None:
    """Pick one option: the choice, its confidence and every option's probability.

    Args:
        question: what to decide.
        options: 2 to 255 options, each `name` or `name=description`.
        state: the content, or @file (.json is structured).
    """
    response = await decide.ask(decide.state(state), {"answer": decide.choice(question, options)})
    render_response(console, response, raw=session.json)


async def score(
    question: str,
    *levels: str,
    state: str,
    decide: Inject[DecideService],
    session: Inject[Session],
    console: Inject[Console],
) -> None:
    """Rate against ordered levels: the expected level, its confidence and the distribution.

    Args:
        question: what to rate.
        levels: 2 to 10 level descriptions, lowest first.
        state: the content, or @file (.json is structured).
    """
    response = await decide.ask(decide.state(state), {"answer": decide.score(question, levels)})
    render_response(console, response, raw=session.json)


async def ask(file: Path, *, decide: Inject[DecideService], session: Inject[Session], console: Inject[Console]) -> None:
    """A full TypeSafe request body from a JSON file: any mix of questions in one call.

    Args:
        file: JSON with `state`, a `questions` map and optionally `model`, as https://docs.typesafe.ai/api documents.
    """
    state, questions, model = decide.request(file)
    render_response(console, await decide.ask(state, questions, model), raw=session.json)


async def listing(*, decide: Inject[DecideService], session: Inject[Session], console: Inject[Console]) -> None:
    """The models the endpoint advertises."""
    render_models(console, await decide.models(), raw=session.json)
