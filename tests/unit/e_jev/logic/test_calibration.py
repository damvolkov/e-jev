import numpy as np
import pytest

from e_jev.logic.calibration import error_calibration, fit_temperature, pad_scores, scale_temperature


async def test_pad_scores_zero_mass_beyond_options() -> None:
    probs = scale_temperature(pad_scores([np.log(np.array([0.7, 0.3])), np.log(np.array([0.2, 0.3, 0.5]))]), 1.0)
    assert probs[0, :2] == pytest.approx([0.7, 0.3])
    assert probs[1, :3] == pytest.approx([0.2, 0.3, 0.5])
    assert probs[:, 3:].sum() == 0


async def test_error_calibration_zero_when_confidence_matches_accuracy() -> None:
    assert error_calibration(np.array([[0.75, 0.25]] * 4), np.array([0, 0, 0, 1])) == pytest.approx(0.0)


async def test_fit_temperature_cools_an_overconfident_model() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 2000)
    logits = rng.normal(0, 1, (2000, 4))
    logits[np.arange(2000), labels] += 1.0
    temperature = fit_temperature(logits * 4, labels)
    assert temperature == pytest.approx(4, rel=0.15)
    assert error_calibration(scale_temperature(logits * 4, temperature), labels) < error_calibration(
        scale_temperature(logits * 4, 1.0), labels
    )
