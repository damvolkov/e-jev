import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

from e_jev.cli.views import describe_answer


@pytest.mark.parametrize(
    ("answer", "kind", "shown"),
    [
        (NoulAnswer(type="noul", noul=0.93), "noul", "yes"),
        (ChoiceAnswer(type="choice", choice="billing", confidence=0.8, probabilities={"billing": 0.9, "sales": 0.1}), "choice", "billing"),
        (
            ScoreAnswer(type="score", score=1.2, confidence=0.7, legend={0: "calm", 1: "angry"}, probabilities={0: 0.1, 1: 0.9}),
            "score",
            "1.20",
        ),
    ],
    ids=["noul", "choice", "score"],
)
async def test_views_describe_answer(answer, kind: str, shown: str) -> None:
    described = describe_answer(answer)
    assert (described[0], described[1].startswith(shown)) == (kind, True)
