"""models.decision: what one Decider answers — raw option scores per question, in log space, with its usage."""

from collections.abc import Mapping
from dataclasses import dataclass

from e_jev.models.reading import Scores


@dataclass(frozen=True, slots=True)
class Decision:
    """Per-question score vectors over the question's options, in order, as one backend produced them.

    Tokens are what the backend billed: the reading path counts them from its prompts, a native server
    answers with the usage its own API reports."""

    scores: Mapping[str, Scores]
    input_tokens: int
    output_tokens: int = 0
