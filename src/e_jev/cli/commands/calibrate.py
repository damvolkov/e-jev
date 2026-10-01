"""cli.commands.calibrate: fit the temperature on labeled JSONL, against the model server itself, and write it.

Runs where the server's settings live (the jev container): it reads vLLM directly, not the HTTP API, so the
scores are the uncalibrated ones the fit needs. One System One question and its truth per line — the choice
key, the score level index, or "true"/"false" for a noul:
    {"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": null, "b": null}}, "label": "a"}
Refit whenever the model or JEV_PERMUTATIONS change; the service refuses a stale fit.
"""

import asyncio
from pathlib import Path

import msgspec
import numpy as np
from rich.console import Console

from e_jev.adapters.reader.vllm import VllmReader
from e_jev.cli.runtime.deps import Inject
from e_jev.core.settings import Settings
from e_jev.logic.calibration import error_calibration, fit_temperature, pad_scores, scale_temperature
from e_jev.logic.primitives import index_label, render_json
from e_jev.models.calibration import Calibration, Labeled
from e_jev.operational.engine import Engine


async def calibrate(file: Path, *, settings: Inject[Settings], console: Inject[Console]) -> None:
    """Fit the temperature on labeled examples and write calibration.json where jev reads it.

    Args:
        file: JSONL, one labeled System One question per line.
    """
    samples = tuple(msgspec.json.Decoder(Labeled).decode_lines(await asyncio.to_thread(file.read_bytes)))
    labels = np.array([index_label(sample.question, sample.label) for sample in samples], dtype=np.intp)
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
    matrix = pad_scores([task.result().scores for task in tasks])
    temperature = fit_temperature(matrix, labels)
    fitted = Calibration(
        model=settings.model,
        permutations=settings.permutations,
        temperature=temperature,
        ece_before=error_calibration(scale_temperature(matrix, 1.0), labels),
        ece_after=error_calibration(scale_temperature(matrix, temperature), labels),
        samples=len(samples),
    )
    settings.calibration.write_bytes(msgspec.json.format(msgspec.json.encode(fitted)))
    console.print(
        f"[green]✓[/] T={fitted.temperature:.3f} · ECE {fitted.ece_before:.3f} → {fitted.ece_after:.3f} · {fitted.samples} samples · {settings.calibration}"
    )
