"""adapters.decider.systemone: one forward to a native `/v1/systemone` server, its probabilities as log scores.

The Laya fixes the plan demands live here by construction: confidence is never taken from the answer —
it is rebuilt from these scores by the engine's `logic.primitives.build_answer` (Jev's formula); a score
level that arrives null in the request was already refused at the boundary (422), so levels reach the
server as text; a choice wider than the backend's budget never reaches this decider — dispatch routed it
away. Capacity answers as the typed leaves; anything out of contract is a gateway error.
"""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Final, Self

import httpx
import msgspec
import numpy as np
import structlog

from e_jev.config.backends import Backend
from e_jev.core.errors import OverloadedError, RateLimitedError, RequestError
from e_jev.models.decision import Decision
from e_jev.models.systemone import Answer, Choice, ChoiceAnswer, NoulAnswer, Question, Score, ScoreAnswer, SystemOneResponse

log = structlog.get_logger()
EPS: Final = 1e-12
ENCODER: Final = msgspec.json.Encoder()
RESPONSE: Final = msgspec.json.Decoder(SystemOneResponse)


class ContractError(RequestError):
    """The server answered, but not for the questions asked, or without a usable distribution."""


def _scores(answer: Answer, question: Question) -> np.ndarray:
    """The answer's probabilities aligned to the question's option order, in log space.

    Servers echo `probabilities` with their own key order — the vector is read BY KEY, never positionally,
    or a choice answered correctly would be scored as its permutation."""
    match answer:
        case NoulAnswer(noul=p):
            probs = np.array([p, 1.0 - p])
        case ChoiceAnswer(probabilities=published) if isinstance(question, Choice):
            probs = np.array([published.get(name, 0.0) for name in question.criteria])
        case ScoreAnswer(probabilities=published) if isinstance(question, Score):
            probs = np.array([published.get(str(level), 0.0) for level in range(len(question.criteria))])
        case ChoiceAnswer(probabilities=published) | ScoreAnswer(probabilities=published):
            probs = np.array(list(published.values()))
    return np.log(np.clip(probs, EPS, None))


class SystemOneDecider:
    __slots__ = ("_client", "_spec")

    def __init__(self, spec: Backend, client: httpx.AsyncClient) -> None:
        self._spec = spec
        self._client = client

    ##### PRIVATE #####

    @staticmethod
    def _open_classify(error: Exception) -> RequestError:
        """The typed upstream failure: capacity answers 429/529, everything else is a gateway error."""
        match error:
            case httpx.HTTPStatusError(response=httpx.Response(status_code=429) as reply):
                hint = (reply.headers.get("retry-after") or "").strip()
                return RateLimitedError(
                    f"rate limited{f', retry in {hint}s' if hint.isdigit() else ''}", float(hint) if hint.isdigit() else None
                )
            case httpx.HTTPStatusError(response=httpx.Response(status_code=503)):
                return OverloadedError("server overloaded")
            case httpx.TimeoutException():
                return OverloadedError("timed out")
            case _:
                return RequestError(str(error))

    ############################################################

    ##### PUBLIC #####

    @classmethod
    @asynccontextmanager
    async def open(cls, spec: Backend, concurrency: int) -> AsyncIterator[Self]:
        """A connected decider for one backend spec, its pool bounded by the fleet's concurrency."""
        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        async with httpx.AsyncClient(
            base_url=spec.url,
            limits=limits,
            timeout=spec.timeout,
            headers={"authorization": f"Bearer {spec.api_key}"},
        ) as client:
            log.info("decider_opened", backend=spec.id, url=spec.url, model=spec.model)
            yield cls(spec, client)

    @property
    def backend(self) -> str:
        return self._spec.id

    async def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        """One request for the whole set: the server reads the state once, every question gets its scores."""
        body: dict[str, Any] = {"model": self._spec.model, "state": state, "questions": questions}
        try:
            reply = await self._client.post("/v1/systemone", content=ENCODER.encode(body))
            reply.raise_for_status()
            response = RESPONSE.decode(reply.content)
        except (httpx.HTTPError, msgspec.DecodeError) as error:
            raise self._open_classify(error) from error
        match sorted(set(questions) - set(response.answers)):
            case [first, *_]:
                message = f"out of contract: no answer for {first}"
                raise ContractError(message)
            case _:
                scores = {name: _scores(response.answers[name], question) for name, question in questions.items()}
        return Decision(scores=scores, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)

    async def health(self) -> bool:
        try:
            reply = await self._client.get("/health")
        except httpx.HTTPError:
            return False
        return reply.is_success
