"""logic.dispatch: which backend may answer which question — shape, language, budget. Pure, no I/O."""

import re
from collections.abc import Mapping, Sequence
from typing import Final

from e_jev.config.backends import Backend, BackendKind, Language
from e_jev.core.errors import UnroutableError
from e_jev.models.systemone import Choice, Noul, Question, QuestionKind, Score

CHARS_PER_TOKEN = 4
ES_MARKS = ("¿", "¡")
ES_STOP = frozenset(
    [
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "que",
        "de",
        "en",
        "con",
        "por",
        "para",
        "se",
        "mi",
        "me",
        "su",
        "sus",
        "es",
        "son",
        "muy",
        "más",
        "pero",
        "como",
        "cuando",
        "dónde",
        "quién",
        "yo",
        "tú",
        "él",
        "ella",
        "nosotros",
        "ustedes",
        "esta",
        "este",
        "está",
        "están",
        "también",
        "dos",
        "tres",
        "año",
        "años",
        "hizo",
        "dijo",
    ]
)
STOPS: Final = 3
STOPS_ACCENTED: Final = 1
WORD = re.compile(r"[a-záéíóúüñ]+")
ACCENT = re.compile(r"[áéíóúüñ¿¡]")


def estimate_tokens(text: str) -> int:
    """Length, not tokenization: dispatch only needs to know which side of a backend's budget a state falls on."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def spanish(text: str) -> bool:
    """Spanish by its strong marks, its accents plus a closed-class word, or three closed-class words alone."""
    words = WORD.findall(text.lower())
    hits = sum(word in ES_STOP for word in words)
    return bool({*text} & {*ES_MARKS}) or hits >= STOPS or (hits >= STOPS_ACCENTED and bool(ACCENT.search(text)))


def option_count(question: Question) -> int:
    match question:
        case Noul():
            return 2
        case Choice(criteria=criteria):
            return len(criteria)
        case Score(criteria=criteria):
            return len(criteria)


def serves(backend: Backend, question: Question, state: str) -> bool:
    """Whether this backend may answer this question at all: language, option budget, state budget.

    The preference *order* is the row's (`config.backends.BackendsSettings`); this is only the gate."""
    wanted = Language.ES if spanish(state) else Language.EN
    allowed = {*backend.languages}
    return (
        (Language.ANY in allowed or wanted in allowed)
        and option_count(question) <= backend.max_options
        and estimate_tokens(state) <= backend.max_state_tokens
    )


def chains(
    rows: Mapping[QuestionKind, Sequence[str]],
    backends: Mapping[str, Backend],
    questions: Mapping[str, Question],
    state: str,
) -> dict[str, list[Backend]]:
    """One ordered fallback chain per question: the row for its kind, eligible backends only — the reader
    always kept as the last resort, a native route that fails or refuses handing the question on to it."""
    out: dict[str, list[Backend]] = {}
    for name, question in questions.items():
        kind = QuestionKind.of(question)
        row = [backends[backend_id] for backend_id in rows.get(kind, ()) if backend_id in backends]
        eligible = [backend for backend in row if serves(backend, question, state)]
        readers = [backend for backend in backends.values() if backend.kind is BackendKind.READING and backend not in eligible]
        chain = [*eligible, *readers]
        match chain:
            case []:
                raise UnroutableError(kind, name, [backend.id for backend in row])
            case _:
                out[name] = chain
    return out
