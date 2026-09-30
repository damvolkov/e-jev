"""Fit the temperature on labeled JSONL and write it where the service reads it.

    python -m e_jev.calibrate labeled.jsonl

One line per example, a System One question plus its truth: the choice key, the score level index,
or "true"/"false" for a noul:
    {"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": null, "b": null}}, "label": "a"}
Refit whenever the model or JEV_PERMUTATIONS change; the service refuses a stale fit.
"""

import asyncio
import sys
from pathlib import Path

import msgspec
import numpy as np
import structlog

from e_jev.calibration import ece, fit, pad, scale
from e_jev.engine import Engine
from e_jev.models import Calibration, Labeled
from e_jev.primitives import label_index, render
from e_jev.settings import Settings

log = structlog.get_logger()


async def calibrate(samples: tuple[Labeled, ...], settings: Settings) -> Calibration:
    engine = await Engine.connect(settings)
    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(engine.read(render(sample.state), sample.question)) for sample in samples]
    finally:
        await engine.aclose()
    matrix = pad([task.result().scores for task in tasks])
    labels = np.array([label_index(sample.question, sample.label) for sample in samples], dtype=np.intp)
    temperature = fit(matrix, labels)
    return Calibration(
        model=settings.model,
        permutations=settings.permutations,
        temperature=temperature,
        ece_before=ece(scale(matrix, 1.0), labels),
        ece_after=ece(scale(matrix, temperature), labels),
        samples=len(samples),
    )


def main() -> None:
    settings = Settings()
    samples = tuple(msgspec.json.Decoder(Labeled).decode_lines(Path(sys.argv[1]).read_bytes()))
    fitted = asyncio.run(calibrate(samples, settings))
    settings.calibration.write_bytes(msgspec.json.format(msgspec.json.encode(fitted)))
    log.info("calibration.written", path=str(settings.calibration), **msgspec.structs.asdict(fitted))


if __name__ == "__main__":
    main()
