from collections.abc import Awaitable, Callable

import msgspec
import pytest

from examples.triage.graph import TriageSettings

pytestmark = pytest.mark.integration


async def test_main_noul_json(ejev: Callable[..., Awaitable[tuple[int, str, str]]], triage_settings: TriageSettings) -> None:
    code, out, _ = await ejev(
        "--url", triage_settings.url, "--json", "noul", "Is this urgent?", "--state", "Checkout is down, we lose orders now."
    )
    assert (code, msgspec.json.decode(out)["answers"]["answer"]["noul"] > 0.5) == (0, True)


async def test_main_choice_table(ejev: Callable[..., Awaitable[tuple[int, str, str]]], triage_settings: TriageSettings) -> None:
    code, out, _ = await ejev(
        "--url", triage_settings.url, "choice", "Which team?", "billing=Charges", "technical=Bugs", "--state", "I was charged twice"
    )
    assert (code, "billing" in out, "request" in out) == (0, True, True)
