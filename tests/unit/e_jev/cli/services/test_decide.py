from pathlib import Path

import pytest

from e_jev.cli.services.decide import DecideInputError, DecideService


async def test_decide_state_text_and_files(cli_resources: Path) -> None:
    assert DecideService.state("as is") == "as is"
    assert DecideService.state(f"@{cli_resources / 'state.txt'}") == "plain text state\n"
    assert DecideService.state(f"@{cli_resources / 'request.json'}")["model"] == "jev-latest"


async def test_decide_choice_descriptions() -> None:
    question = DecideService.choice("Team?", ("billing=Payments", "technical", "sales=Pricing, upgrades"))
    assert question.criteria == {"billing": "Payments", "technical": None, "sales": "Pricing, upgrades"}


async def test_decide_noul_criteria_only_when_given() -> None:
    assert (DecideService.noul("Urgent?").criteria, DecideService.noul("Urgent?", true="now").criteria is not None) == (None, True)


async def test_decide_request(cli_resources: Path) -> None:
    state, questions, model = DecideService.request(cli_resources / "request.json")
    assert (sorted(questions), model, state["message"][:5]) == (["is_urgent"], "jev-latest", "Help!")


async def test_decide_request_rejects_a_bare_state(cli_resources: Path) -> None:
    with pytest.raises(DecideInputError):
        DecideService.request(cli_resources / "state.txt")
