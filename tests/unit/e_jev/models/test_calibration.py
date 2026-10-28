import msgspec
import pytest

from e_jev.core.settings import Settings
from e_jev.models.calibration import Calibration, KindFit
from e_jev.models.systemone import Choice, Noul, Question, QuestionKind, Score


def _fit(model: str = "m") -> KindFit:
    return KindFit(
        model=model,
        temperatures={QuestionKind.NOUL: 1.7},
        ece_before={QuestionKind.NOUL: 0.1},
        ece_after={QuestionKind.NOUL: 0.02},
        accuracy={"boolq": 0.8},
        fitted=150,
        held_out=150,
    )


@pytest.mark.parametrize(("permutations", "applied"), [(1, True), (2, False)])
async def test_calibration_load_fingerprint(settings: Settings, permutations: int, applied: bool) -> None:
    fitted = Calibration(permutations=1, backends={"reading": _fit()})
    settings.calibration.write_bytes(msgspec.json.encode(fitted))
    assert (Calibration.load(settings.calibration, permutations) == fitted) is applied


async def test_calibration_load_missing(settings: Settings) -> None:
    assert Calibration.load(settings.calibration, 1) is None


async def test_calibration_temperature_answers_per_backend_and_model(settings: Settings) -> None:
    fitted = Calibration(permutations=1, backends={"reading": _fit("m"), "clef": _fit("clef")})
    assert fitted.temperature("reading", "m", QuestionKind.NOUL) == 1.7
    assert fitted.temperature("jevos", "m", QuestionKind.NOUL) == 1.0
    assert fitted.temperature("reading", "m", QuestionKind.SCORE) == 1.0
    assert fitted.temperature("reading", "other-model", QuestionKind.NOUL) == 1.0


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


async def test_calibration_load_unreadable_legacy_file(settings: Settings) -> None:
    settings.calibration.write_bytes(
        b'{"permutations":1,"backends":{"reading":{"temperatures":{},"ece_before":{},"ece_after":{},"accuracy":{},"fitted":0,"held_out":0}}}'
    )
    assert Calibration.load(settings.calibration, 1) is None
