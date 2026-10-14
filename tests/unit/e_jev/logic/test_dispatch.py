"""logic.dispatch: which backend may answer which question — shape, language, budget. Pure, no I/O."""

import pytest

from e_jev.config.backends import Backend, BackendKind, Language
from e_jev.logic.dispatch import chains, estimate_tokens, serves, spanish
from e_jev.models.systemone import Choice, Noul, QuestionKind, Score

READING = Backend(id="reading", kind=BackendKind.READING, languages=(Language.ANY,))
CLEF = Backend(id="clef", kind=BackendKind.SYSTEMONE, url="http://x/v1", languages=(Language.EN,))
LAYA_MULTI = Backend(
    id="laya", kind=BackendKind.LAYA, url="http://y/v1", languages=(Language.EN, Language.ES), max_options=20, max_state_tokens=4096
)

ES_STATE = "¿Qué hago? El pedido llegó roto y quiero un reembolso de los dos cobros de marzo."
EN_STATE = "The order arrived broken. I was charged twice for March and want a refund."
WIDE = Choice(criteria=dict.fromkeys(f"o{i}" for i in range(30)))
SMALL = Choice(criteria={"a": None, "b": None})
NOUL = Noul(instructions="?")
SCORE = Score(criteria=["low", "high"])


@pytest.mark.parametrize(("state", "expected"), [(ES_STATE, True), (EN_STATE, False), ("", False), ("¿?", True)])
def test_dispatch_spanish_detection(state: str, expected: bool) -> None:
    assert spanish(state) is expected


def test_dispatch_estimate_tokens_scales_with_length() -> None:
    assert estimate_tokens("a" * 4000) > estimate_tokens("a" * 40)
    assert estimate_tokens(EN_STATE) < 64


def test_dispatch_serves_language_and_shape() -> None:
    assert serves(CLEF, SMALL, EN_STATE)
    assert not serves(CLEF, SMALL, ES_STATE)
    assert serves(LAYA_MULTI, SMALL, ES_STATE)
    assert not serves(LAYA_MULTI, WIDE, ES_STATE)
    assert not serves(LAYA_MULTI, SMALL, ES_STATE + " x" * 20_000)
    assert serves(READING, WIDE, ES_STATE)


def test_dispatch_chains_order_by_row_and_end_in_reading() -> None:
    rows = {QuestionKind.CHOICE: ["laya", "clef"], QuestionKind.CHOICE_WIDE: ["clef"]}
    backends = {b.id: b for b in (READING, CLEF, LAYA_MULTI)}
    questions = {"s": SMALL, "w": WIDE, "n": NOUL}
    found = chains(rows, backends, questions, EN_STATE)
    assert found["s"] == [LAYA_MULTI, CLEF, READING]
    assert found["w"] == [CLEF, READING]
    assert [b.id for b in found["n"]] == ["reading", "clef", "laya"]


def test_dispatch_chains_spanish_drops_english_only() -> None:
    rows = {QuestionKind.NOUL: ["clef", "laya", "reading"]}
    backends = {b.id: b for b in (READING, CLEF, LAYA_MULTI)}
    found = chains(rows, backends, {"q": NOUL}, ES_STATE)
    assert [b.id for b in found["q"]] == ["laya", "reading"]
