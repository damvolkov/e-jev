import pytest

from e_jev.logic.usage import count_tokens


@pytest.mark.parametrize(
    ("prompts", "expected"),
    [([(1, 2, 3)], 3), ([(1, 2, 3, 4), (1, 2, 3, 5, 6)], 3 + 1 + 2), ([(1, 2), (3, 4)], 4), ([], 0)],
    ids=["one", "shared-state", "nothing-shared", "empty"],
)
async def test_count_tokens(prompts: list[tuple[int, ...]], expected: int) -> None:
    assert count_tokens(prompts) == expected
