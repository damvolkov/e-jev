import numpy as np
import pytest

from e_jev.calibration import ece, fit, pad, scale


def test_pad_zero_mass_beyond_options() -> None:
    probs = scale(pad([np.log(np.array([0.7, 0.3])), np.log(np.array([0.2, 0.3, 0.5]))]), 1.0)
    assert probs[0, :2] == pytest.approx([0.7, 0.3])
    assert probs[1, :3] == pytest.approx([0.2, 0.3, 0.5])
    assert probs[:, 3:].sum() == 0


def test_ece_zero_when_confidence_matches_accuracy() -> None:
    probs = np.array([[0.75, 0.25]] * 4)
    assert ece(probs, np.array([0, 0, 0, 1])) == pytest.approx(0.0)


def test_fit_cools_an_overconfident_model() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, 2000)
    logits = rng.normal(0, 1, (2000, 4))
    logits[np.arange(2000), labels] += 1.0
    overconfident = logits * 4
    temperature = fit(overconfident, labels)
    assert temperature == pytest.approx(4, rel=0.15)
    assert ece(scale(overconfident, temperature), labels) < ece(scale(overconfident, 1.0), labels)
