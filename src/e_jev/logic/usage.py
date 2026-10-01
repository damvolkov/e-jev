"""logic.usage: input tokens as TypeSafe bills them — the shared state once, each question's own tokens apart. Pure."""

from collections.abc import Sequence
from os.path import commonprefix


def count_tokens(prompts: Sequence[Sequence[int]]) -> int:
    """The longest common prefix once, plus what each prompt adds beyond it: what a prefix cache computes."""
    shared = len(commonprefix([list(prompt) for prompt in prompts]))
    return shared + sum(len(prompt) - shared for prompt in prompts)
