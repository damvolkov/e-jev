"""cli.commands.calibrate: fit one temperature per question kind on labeled JSONL, against the model server itself.

Runs where the server's settings live: it reads vLLM directly, not the HTTP API, so the scores are the
uncalibrated ones the fit needs. Each example falls in a stable half — fit or held out — and every number
it reports (ECE before and after, accuracy per source) comes from the held-out half. One labeled System One
question per line: the choice key, the score level index, or "true"/"false" for a noul:
    {"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": null}}, "label": "a", "source": "x"}
Refit whenever the model or JEV_PERMUTATIONS change; the service refuses a stale fit.
"""

import asyncio
from pathlib import Path

import msgspec
from rich.console import Console
from rich.table import Table

from e_jev.adapters.reader.vllm import VllmReader
from e_jev.cli.runtime.deps import Inject
from e_jev.cli.views import HEADER
from e_jev.core.settings import Settings
from e_jev.logic.calibration import Scored, fit_kinds, split_held
from e_jev.logic.primitives import index_label, render_json
from e_jev.models.calibration import Calibration, Labeled
from e_jev.models.systemone import QuestionKind
from e_jev.operational.engine import Engine


async def calibrate(file: Path, *, settings: Inject[Settings], console: Inject[Console]) -> None:
    """Fit per-kind temperatures on half the examples, measure on the other half, write calibration.json.

    Args:
        file: JSONL, one labeled System One question per line.
    """
    samples = tuple(msgspec.json.Decoder(Labeled).decode_lines(await asyncio.to_thread(file.read_bytes)))
    async with VllmReader.open(settings) as reader:
        engine = Engine(
            reader,
            model=settings.model,
            permutations=settings.permutations,
            concurrency=settings.concurrency,
            calibration=None,
            epsilon=settings.trie_epsilon,
        )
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(engine.read(render_json(sample.state), sample.question)) for sample in samples]
    rows = [
        Scored(
            kind=QuestionKind.of(sample.question),
            source=sample.source,
            scores=task.result().scores,
            label=index_label(sample.question, sample.label),
            held=split_held(msgspec.json.encode([sample.state, sample.question])),
        )
        for sample, task in zip(samples, tasks, strict=True)
    ]
    fit = fit_kinds(rows)
    fitted = Calibration(
        model=settings.model,
        permutations=settings.permutations,
        temperatures={QuestionKind(kind): value for kind, value in fit.temperatures.items()},
        ece_before={QuestionKind(kind): value for kind, value in fit.ece_before.items()},
        ece_after={QuestionKind(kind): value for kind, value in fit.ece_after.items()},
        accuracy=fit.accuracy,
        fitted=sum(not row.held for row in rows),
        held_out=sum(row.held for row in rows),
    )
    await asyncio.to_thread(settings.calibration.write_bytes, msgspec.json.format(msgspec.json.encode(fitted)))
    table = Table("kind", "temperature", "ECE before", "ECE after", box=None, header_style=HEADER)
    for kind, temperature in fitted.temperatures.items():
        table.add_row(kind, f"{temperature:.3f}", f"{fitted.ece_before[kind]:.3f}", f"{fitted.ece_after[kind]:.3f}")
    console.print(table)
    console.print(", ".join(f"{source} {value:.1%}" for source, value in fitted.accuracy.items()))
    console.print(f"[green]✓[/] fitted on {fitted.fitted}, measured on {fitted.held_out} held out · {settings.calibration}")
