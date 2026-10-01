import numpy as np
import pytest

from e_jev.logic.calibration import Scored, error_calibration, fit_kinds, fit_temperature, pad_scores, scale_temperature, split_held


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


async def test_split_held_is_stable_and_balanced() -> None:
    keys = [f"example-{index}".encode() for index in range(2000)]
    assert [split_held(key) for key in keys] == [split_held(key) for key in keys]
    assert 0.45 < np.mean([split_held(key) for key in keys]) < 0.55


async def test_fit_kinds_cools_each_kind_on_its_own() -> None:
    rng = np.random.default_rng(1)
    rows = [
        Scored(kind=kind, source=f"{kind}-set", scores=logits * scale, label=int(label), held=bool(index % 2))
        for kind, scale in (("noul", 3.0), ("choice", 1.0))
        for index, label in enumerate(rng.integers(0, 2, 1200))
        for logits in (rng.normal(0, 1, 2) + np.eye(2)[label],)
    ]
    fit = fit_kinds(rows)
    assert fit.temperatures["noul"] > 2 * fit.temperatures["choice"]
    assert fit.ece_after["noul"] < fit.ece_before["noul"]
    assert set(fit.accuracy) == {"noul-set", "choice-set"}


async def test_pad_scores_beyond_the_letters() -> None:
    matrix = pad_scores([np.zeros(77), np.zeros(2)])
    assert (matrix.shape, np.isinf(matrix[1, 2:]).all()) == ((2, 77), True)
