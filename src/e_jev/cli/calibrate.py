"""cli.calibrate: fit the temperature on labeled JSONL and write it where the service reads it.

    python -m e_jev.cli.calibrate labeled.jsonl

One System One question and its truth per line — the choice key, the score level index, or
"true"/"false" for a noul:
    {"state": "...", "question": {"type": "choice", "instructions": "...", "criteria": {"a": null, "b": null}}, "label": "a"}
Refit whenever the model or JEV_PERMUTATIONS change; the service refuses a stale fit.
"""

import asyncio
import sys
from pathlib import Path

import msgspec
import numpy as np
import structlog

from e_jev.adapters.reader.vllm import VllmReader
from e_jev.core.logger import setup_logger
from e_jev.core.settings import Settings
from e_jev.core.settings import settings as st
from e_jev.logic.calibration import error_calibration, fit_temperature, pad_scores, scale_temperature
from e_jev.logic.primitives import index_label, render_json
from e_jev.models.calibration import Calibration, Labeled
from e_jev.operational.engine import Engine

log = structlog.get_logger()


async def calibrate(samples: tuple[Labeled, ...], settings: Settings) -> Calibration:
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
    return Calibration(
        model=settings.model,
        permutations=settings.permutations,
        temperature=temperature,
        ece_before=error_calibration(scale_temperature(matrix, 1.0), labels),
        ece_after=error_calibration(scale_temperature(matrix, temperature), labels),
        samples=len(samples),
    )


def main() -> None:
    setup_logger(st.env, st.log_level)
    samples = tuple(msgspec.json.Decoder(Labeled).decode_lines(Path(sys.argv[1]).read_bytes()))
    fitted = asyncio.run(calibrate(samples, st))
    st.calibration.write_bytes(msgspec.json.format(msgspec.json.encode(fitted)))
    log.info("calibration_written", path=str(st.calibration), **msgspec.structs.asdict(fitted))


if __name__ == "__main__":
    main()
