"""scripts/backend_bench.py: measure one decision backend against the TypeSafe contract.

    uv run python scripts/backend_bench.py CONFIG BASE_URL --file SET.jsonl [--file OTHER.jsonl ...]

For every labeled row it sends one single-question System One request, recovers the option scores from the
answer probabilities, fits per-kind temperatures on the stable fit half and reports accuracy and ECE on the
held-out half — `ejev calibrate`'s rule at the request level, so any backend reachable through the contract
is measured identically. Latency runs on a sampled slice: sequential, then at concurrency 16 and 32.

Raw scores: a decision server answers with calibrated probabilities, not logprobs — "ECE before" is the ECE
of what the server returned; e-jev's own fit rides on top of it.
"""

import argparse
import asyncio
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import msgspec
import numpy as np

from e_jev.logic.calibration import Scored, fit_kinds, split_held
from e_jev.logic.primitives import index_label
from e_jev.models.calibration import Labeled
from e_jev.models.systemone import (
    Answer,
    Choice,
    ChoiceAnswer,
    NoulAnswer,
    Question,
    QuestionKind,
    Score,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)

EPS = 1e-12
ENCODER = msgspec.json.Encoder()
RESPONSE = msgspec.json.Decoder(SystemOneResponse)


def scores_of(answer: Answer, question: Question) -> np.ndarray:
    """The option scores behind one answer: published probabilities, aligned BY KEY to the question, in log space."""
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


class Backend:
    __slots__ = ("_base", "_client", "_model", "failures")

    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self._base = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout, headers={"authorization": "Bearer local", "content-type": "application/json"})
        self.failures: list[dict[str, Any]] = []

    async def _open_answer(self, sample: Labeled) -> tuple[Answer, float]:
        start = time.perf_counter()
        request = SystemOneRequest(state=sample.state, questions={"q": sample.question}, model=self._model)
        try:
            reply = await self._client.post(f"{self._base}/v1/systemone", content=ENCODER.encode(request))
            reply.raise_for_status()
            answer = RESPONSE.decode(reply.content).answers["q"]
        except (httpx.HTTPError, msgspec.DecodeError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            self.failures.append({"source": sample.source, "status": status, "error": str(error)[:200]})
            raise
        return answer, time.perf_counter() - start

    async def read_all(self, samples: Sequence[Labeled], concurrency: int) -> list[tuple[Labeled, Answer] | None]:
        sem = asyncio.Semaphore(concurrency)

        async def one(index: int, sample: Labeled) -> tuple[int, tuple[Labeled, Answer] | None]:
            async with sem:
                try:
                    answer, _ = await self._open_answer(sample)
                except (httpx.HTTPError, msgspec.DecodeError):
                    return index, None
                return index, (sample, answer)

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(index, sample)) for index, sample in enumerate(samples)]
        done = dict(task.result() for task in tasks)
        return [done.get(index) for index in range(len(samples))]

    async def latency(self, samples: Sequence[Labeled], concurrencies: Sequence[int]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for conc in concurrencies:
            sem = asyncio.Semaphore(conc)

            async def one(sample: Labeled, gate: asyncio.Semaphore = sem) -> tuple[str, float] | None:
                async with gate:
                    try:
                        _, seconds = await self._open_answer(sample)
                    except (httpx.HTTPError, msgspec.DecodeError):
                        return None
                    return sample.source, seconds

            started = time.perf_counter()
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(one(sample)) for sample in samples]
            wall = time.perf_counter() - started
            times = [value for value in (task.result() for task in tasks) if value is not None]
            values = [value for _, value in times]
            sources = sorted({source for source, _ in times})
            out[f"c{conc}"] = {
                "p50": float(np.percentile(values, 50)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
                "per_source": {
                    source: percentile([value for item, value in times if item == source]) for source in sources
                },
                "throughput": len(times) / wall,
            }
        return out

    async def close(self) -> None:
        await self._client.aclose()


def scored_rows(answers: Sequence[tuple[Labeled, Answer] | None]) -> list[Scored]:
    """Labeled rows that answered, in the calibration shape: kind, raw scores, true option, split."""
    return [
        Scored(
            kind=QuestionKind.of(sample.question),
            source=sample.source,
            scores=scores_of(answer, sample.question),
            label=index_label(sample.question, sample.label),
            held=split_held(msgspec.json.encode([sample.state, sample.question])),
        )
        for sample, answer in (pair for pair in answers if pair is not None)
    ]


def percentile(times: Sequence[float]) -> dict[str, float | int | None]:
    match list(times):
        case []:
            return {"p50": None, "p95": None, "n": 0}
        case values:
            return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "n": len(values)}


def sample_slice(pairs: Sequence[tuple[Labeled, Answer]], per_kind: int) -> list[Labeled]:
    """Up to `per_kind` answered samples per kind, shortest states first: latency, not token count."""
    by_kind: dict[str, list[Labeled]] = {}
    for sample, _ in pairs:
        by_kind.setdefault(QuestionKind.of(sample.question), []).append(sample)
    return [
        picked
        for chosen in by_kind.values()
        for picked in sorted(chosen, key=lambda sample: len(msgspec.json.encode(sample.state)))[:per_kind]
    ]


async def run(config: str, base_url: str, files: Sequence[Path], args: argparse.Namespace) -> dict[str, Any]:
    samples = [sample for file in files for sample in msgspec.json.Decoder(Labeled).decode_lines(await asyncio.to_thread(file.read_bytes))]
    backend = Backend(base_url, args.model, args.timeout)
    try:
        answers = await backend.read_all(samples, args.concurrency)
        fit = fit_kinds(scored_rows(answers))
        pairs = [pair for pair in answers if pair is not None]
        phase = await backend.latency(sample_slice(pairs, args.lat_per_kind), [1, *args.concurrencies])
        return {
            "config": config,
            "base_url": base_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model_requested": args.model,
            "rows": {"total": len(samples), "answered": len(pairs), "failed": len(samples) - len(pairs)},
            "fit": {
                "temperatures": dict(fit.temperatures),
                "ece_before": dict(fit.ece_before),
                "ece_after": dict(fit.ece_after),
                "accuracy": fit.accuracy,
            },
            "latency": phase,
            "failures": backend.failures[:50],
        }
    finally:
        await backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("config")
    parser.add_argument("base_url")
    parser.add_argument("--file", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--concurrencies", type=lambda text: [int(value) for value in text.split(",")], default=[16, 32])
    parser.add_argument("--lat-per-kind", type=int, default=40)
    args = parser.parse_args()
    report = asyncio.run(run(args.config, args.base_url, args.file, args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(msgspec.json.format(msgspec.json.encode(report)))
    print(f"{args.config}: {report['fit']['accuracy']} · ECE after {report['fit']['ece_after']}")


if __name__ == "__main__":
    main()
