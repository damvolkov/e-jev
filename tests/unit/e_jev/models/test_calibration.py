import msgspec
import pytest

from e_jev.core.settings import Settings
from e_jev.models.calibration import Calibration


@pytest.mark.parametrize(("model", "permutations", "applied"), [("m", 1, True), ("other", 1, False), ("m", 2, False)])
async def test_calibration_load_fingerprint(settings: Settings, model: str, permutations: int, applied: bool) -> None:
    fitted = Calibration(model=model, permutations=permutations, temperature=1.7, ece_before=0.1, ece_after=0.02, samples=300)
    settings.calibration.write_bytes(msgspec.json.encode(fitted))
    assert (Calibration.load(settings.calibration, "m", 1) == fitted) is applied


async def test_calibration_load_missing(settings: Settings) -> None:
    assert Calibration.load(settings.calibration, "m", 1) is None
