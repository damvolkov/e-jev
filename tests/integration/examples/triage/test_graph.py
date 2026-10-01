import pytest

from examples.triage.graph import Area, Priority, TriageSettings, build_triage, triage

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("ticket", "queue", "page"),
    [
        ("BUY CHEAP REPLICA WATCHES!!! 90% OFF, click www.watches-cheap.biz now", "spam", False),
        ("Since 10 minutes your public API returns HTTP 500 on every call and our checkout is down. Fix it now.", "api", True),
        ("I was charged twice for the Pro plan this month, please refund one of the charges.", "billing", False),
        ("How do I export a dashboard to PDF?", "exports", False),
    ],
    ids=["spam", "outage", "double-charge", "how-to"],
)
async def test_graph_triage_routes(triage_settings: TriageSettings, ticket: str, queue: str, page: bool) -> None:
    verdict, state = await triage(ticket, triage_settings)
    assert (verdict.queue, verdict.page) == (queue, page)
    assert all(0.0 <= value <= 1.0 for value in state.confidence.values())


async def test_graph_triage_reads_a_choice_past_the_letters(triage_settings: TriageSettings) -> None:
    _, state = await triage("Our SAML login with Okta fails with an invalid audience error for every user.", triage_settings)
    assert state.classification is not None
    assert (len(Area) > 26, state.classification.area, state.classification.priority >= Priority.NORMAL) == (True, Area.SSO, True)


async def test_graph_triage_renders_mermaid() -> None:
    mermaid = build_triage().render(title="triage")
    assert all(edge in mermaid for edge in ("decision --> discard: spam", "decision --> classify: customer", "route --> [*]"))
