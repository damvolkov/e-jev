import numpy as np
import pytest
from scipy.special import logsumexp

from e_jev.logic.labels import LETTERS, NUMBERS, frontier_labels, normalize_labels, plan_labels, prune_labels, score_labels, select_labels

STOP = 0


def digits(label: str) -> tuple[int, ...]:
    return tuple(map(ord, label))


@pytest.mark.parametrize(("size", "first", "last"), [(2, "A", "B"), (26, "A", "Z"), (27, "1", "27"), (255, "1", "255")])
async def test_select_labels(size: int, first: str, last: str) -> None:
    labels = select_labels(size)
    assert (len(labels), labels[0], labels[-1]) == (size, first, last)


@pytest.mark.parametrize(("labels", "passes"), [(LETTERS, 1), (NUMBERS[:9], 1), (NUMBERS[:27], 3), (NUMBERS, 26)])
async def test_plan_labels_passes(labels: tuple[str, ...], passes: int) -> None:
    assert len(plan_labels([digits(label) for label in labels], STOP)) == passes


@pytest.mark.parametrize("labels", [LETTERS[:3], NUMBERS[:40], NUMBERS])
async def test_score_labels_is_a_distribution(labels: tuple[str, ...]) -> None:
    rng = np.random.default_rng(7)
    sequences = [digits(label) for label in labels]
    nodes = plan_labels(sequences, STOP)
    answers = {node.prefix: dict(zip(node.candidates, rng.normal(0, 3, len(node.candidates)).tolist(), strict=True)) for node in nodes}
    assert float(np.exp(logsumexp(score_labels(sequences, STOP, nodes, answers)))) == pytest.approx(1.0)


async def test_score_labels_chain_rule() -> None:
    sequences = [digits("1"), digits("12"), digits("2")]
    nodes = plan_labels(sequences, STOP)
    root, one = (ord("1"), ord("2")), (STOP, ord("2"))
    assert [(node.prefix, node.candidates) for node in nodes] == [((), root), ((ord("1"),), one)]
    answers: dict[tuple[int, ...], dict[int, float]] = {
        (): {ord("1"): float(np.log(0.6)), ord("2"): float(np.log(0.4))},
        (ord("1"),): {STOP: float(np.log(0.25)), ord("2"): float(np.log(0.75))},
    }
    assert np.exp(score_labels(sequences, STOP, nodes, answers)) == pytest.approx([0.15, 0.45, 0.4])


@pytest.mark.parametrize(("budget", "kept", "dropped"), [(0.0, 3, 0.0), (0.05, 2, 0.04), (0.5, 1, 0.3)])
async def test_prune_labels_drops_lightest_within_budget(budget: float, kept: int, dropped: float) -> None:
    frontier: dict[tuple[int, ...], float] = {(1,): 0.7, (2,): 0.26, (3,): 0.04}
    prefixes, mass = prune_labels(frontier, budget)
    assert (len(prefixes), prefixes[-1], mass) == (kept, (1,), pytest.approx(dropped))


async def test_frontier_labels_carries_mass() -> None:
    sequences = [digits("1"), digits("12"), digits("2"), digits("21")]
    nodes = plan_labels(sequences, STOP)
    root = {ord("1"): float(np.log(0.75)), ord("2"): float(np.log(0.25))}
    assert frontier_labels(nodes, {(): 1.0}, {(): root}) == pytest.approx({(ord("1"),): 0.75, (ord("2"),): 0.25})


@pytest.mark.parametrize("epsilon", [1e-4, 1e-2])
async def test_pruned_read_within_tv_bound(epsilon: float) -> None:
    rng = np.random.default_rng(3)
    sequences = [digits(label) for label in NUMBERS]
    nodes = plan_labels(sequences, STOP)
    raw = {
        node.prefix: normalize_labels(dict(zip(node.candidates, rng.normal(0, 4, len(node.candidates)).tolist(), strict=True)))
        for node in nodes
    }
    exact = np.exp(score_labels(sequences, STOP, nodes, raw))
    level: dict[tuple[int, ...], float] = {(): 1.0}
    read: dict[tuple[int, ...], dict[int, float]] = {}
    budget = epsilon
    for _ in range(4):
        kept, dropped = prune_labels(level, budget)
        budget -= dropped
        read |= {prefix: raw[prefix] for prefix in kept}
        level = frontier_labels(nodes, {prefix: level[prefix] for prefix in kept}, read)
    approx = np.exp(score_labels(sequences, STOP, nodes, read))
    assert (approx.sum(), 0.5 * np.abs(exact - approx).sum() <= epsilon) == (pytest.approx(1.0), True)
