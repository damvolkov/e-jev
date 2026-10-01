"""logic.labels: exact option probabilities for multi-token labels, walked as a token trie. Pure, no I/O.

Every label is a token sequence closed by the stop token. A trie node whose continuations number two or
more is one forward pass: the next-token logprobs restricted to those continuations, renormalized. A
label's log-probability is the sum of its edges, so the distribution over labels is proper by
construction (chain rule), and nothing outside the labels can win. Single-token labels (letters) are a
root-only trie: one pass. Numbers 1..255 span 26 nodes.

Nodes are expanded best-first, wave by wave: the lightest frontier nodes whose masses add up to less
than the budget ε stay unread, and their continuations count as uniform. The distribution stays proper
and its total-variation distance to the exact one is at most ε; ε = 0 reads every node.
"""

import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import Final

import numpy as np
from scipy.special import logsumexp

from e_jev.models.reading import Scores

type Tokens = tuple[int, ...]

LETTERS: Final = tuple(string.ascii_uppercase)
NUMBERS: Final = tuple(str(number) for number in range(1, 256))
UNIVERSE: Final = (*LETTERS, *NUMBERS)


@dataclass(frozen=True, slots=True)
class Node:
    """One forward pass: the tokens after the prompt, and the continuations it chooses between."""

    prefix: Tokens
    candidates: Tokens


def select_labels(size: int) -> tuple[str, ...]:
    """Letters while they last — the format every instruct model knows — then numbers."""
    return LETTERS[:size] if size <= len(LETTERS) else NUMBERS[:size]


def plan_labels(sequences: Sequence[Tokens], stop: int) -> tuple[Node, ...]:
    """The trie nodes that need a forward pass: those with two or more continuations."""
    edges = sorted(
        {(closed[:depth], closed[depth]) for sequence in sequences for closed in ((*sequence, stop),) for depth in range(len(closed))}
    )
    grouped = ((prefix, tuple(token for _, token in group)) for prefix, group in groupby(edges, key=itemgetter(0)))
    return tuple(Node(prefix, candidates) for prefix, candidates in grouped if len(candidates) > 1)


def normalize_labels(logprobs: Mapping[int, float]) -> dict[int, float]:
    """One node's raw logprobs renormalized over its candidates."""
    total = logsumexp(list(logprobs.values()))
    return {token: value - total for token, value in logprobs.items()}


def frontier_labels(
    nodes: Sequence[Node], parents: Mapping[Tokens, float], answers: Mapping[Tokens, Mapping[int, float]]
) -> dict[Tokens, float]:
    """The nodes one level below the expanded `parents`, each with its mass: parent mass times edge probability."""
    return {
        node.prefix: parents[node.prefix[:-1]] * float(np.exp(answers[node.prefix[:-1]][node.prefix[-1]]))
        for node in nodes
        if node.prefix and node.prefix[:-1] in parents
    }


def prune_labels(frontier: Mapping[Tokens, float], budget: float) -> tuple[tuple[Tokens, ...], float]:
    """The nodes worth a pass, lightest dropped while their summed mass stays within budget; and the mass dropped."""
    ordered = sorted(frontier.items(), key=itemgetter(1))
    cumulative = np.cumsum([mass for _, mass in ordered])
    dropped = int(np.searchsorted(cumulative, budget, side="right"))
    return tuple(prefix for prefix, _ in ordered[dropped:]), float(cumulative[dropped - 1]) if dropped else 0.0


def score_labels(sequences: Sequence[Tokens], stop: int, nodes: Sequence[Node], answers: Mapping[Tokens, Mapping[int, float]]) -> Scores:
    """Log-probability of every label: renormalized edges summed along its path; unread nodes uniform, forced edges 0."""
    uniform = {node.prefix: dict.fromkeys(node.candidates, -float(np.log(len(node.candidates)))) for node in nodes}
    normalized = uniform | {prefix: normalize_labels(logprobs) for prefix, logprobs in answers.items()}
    return np.array(
        [
            sum(normalized.get(closed[:depth], {}).get(closed[depth], 0.0) for depth in range(len(closed)))
            for sequence in sequences
            for closed in ((*sequence, stop),)
        ]
    )
