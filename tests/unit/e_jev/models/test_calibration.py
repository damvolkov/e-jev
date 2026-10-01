import msgspec
import pytest

from e_jev.core.settings import Settings
from e_jev.models.calibration import Calibration
from e_jev.models.systemone import Choice, Noul, Question, QuestionKind, Score


@pytest.mark.parametrize(("model", "permutations", "applied"), [("m", 1, True), ("other", 1, False), ("m", 2, False)])
async def test_calibration_load_fingerprint(settings: Settings, model: str, permutations: int, applied: bool) -> None:
    fitted = Calibration(
        model=model,
        permutations=permutations,
        temperatures={QuestionKind.NOUL: 1.7},
        ece_before={QuestionKind.NOUL: 0.1},
        ece_after={QuestionKind.NOUL: 0.02},
        accuracy={"boolq": 0.8},
        fitted=150,
        held_out=150,
    )
    settings.calibration.write_bytes(msgspec.json.encode(fitted))
    assert (Calibration.load(settings.calibration, "m", 1) == fitted) is applied


async def test_calibration_load_missing(settings: Settings) -> None:
    assert Calibration.load(settings.calibration, "m", 1) is None


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        (Noul(instructions="?"), QuestionKind.NOUL),
        (Choice(criteria=dict.fromkeys("ab")), QuestionKind.CHOICE),
        (Choice(criteria={str(index): None for index in range(27)}), QuestionKind.CHOICE_WIDE),
        (Score(criteria=["low", "high"]), QuestionKind.SCORE),
    ],
    ids=["noul", "choice", "choice-wide", "score"],
)
async def test_question_kind_follows_the_readout(question: Question, kind: QuestionKind) -> None:
    assert QuestionKind.of(question) == kind
