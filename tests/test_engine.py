import msgspec
import numpy as np
import pytest

from e_jev.engine import Engine, orders, prompt
from e_jev.models import Calibration
from e_jev.primitives import Readout
from e_jev.settings import Settings


def test_prompt_letters_follow_order() -> None:
    question = Readout(task="Pick.", instructions="q?", options=("yes", "no"))
    assert prompt("s", question, (1, 0)).endswith("Options:\nA. no\nB. yes")


@pytest.mark.parametrize(("permutations", "expected"), [(1, ((0, 1, 2),)), (2, ((0, 1, 2), (2, 1, 0)))])
def test_orders(permutations: int, expected: tuple[tuple[int, ...], ...]) -> None:
    assert orders(3, permutations) == expected


def test_reorder_restores_original_option_order() -> None:
    order = (2, 0, 1)
    shown = np.array([30.0, 10.0, 20.0])
    assert shown[np.argsort(order)].tolist() == [10.0, 20.0, 30.0]


@pytest.mark.parametrize(("model", "permutations", "applied"), [("m", 1, True), ("other", 1, False), ("m", 2, False)])
def test_calibration_applies_only_to_its_fingerprint(tmp_path, model: str, permutations: int, applied: bool) -> None:
    path = tmp_path / "calibration.json"
    fitted = Calibration(model=model, permutations=permutations, temperature=1.7, ece_before=0.1, ece_after=0.02, samples=300)
    path.write_bytes(msgspec.json.encode(fitted))
    assert (Engine.load(Settings(model="m", permutations=1, calibration=path)) == fitted) is applied


def test_calibration_missing(tmp_path) -> None:
    assert Engine.load(Settings(calibration=tmp_path / "none.json")) is None
