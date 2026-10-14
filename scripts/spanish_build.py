"""scripts/spanish_build.py: a labeled Spanish set, one System One kind each, mirroring calibration_build.

    uv run python scripts/spanish_build.py OUT.jsonl [PER_SOURCE]

MASSIVE es-ES intents → choice (60, wide) and choice (20, the size Laya evaluates — gold + 19 sampled
distractors) · cardiffnlp tweet_sentiment_multilingual spanish → score (3) · facebook/xnli spanish → noul.
Ids and revisions (2026-10-13): AmazonScience/massive ff6bd8e data-version 1.0 · cardiffnlp/tweet_sentiment_multilingual
40f65d4 · facebook/xnli b8dd5d7. MASSIVE 1.0 carries no sentiment, hence the tweet set for the score kind.
"""

import random
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import msgspec
import numpy as np

SEED = 7
DEFAULT_SIZE = 300
MASSIVE_TGZ = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
ROWS_API = "https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}&split={split}&offset={offset}&length={length}"
INTENTS_N = 19
SENTIMENT = ["negative", "neutral", "positive"]
TRUE_GIVEN = "Is the following statement true given the text above?"


def massive_es(cache: Path) -> list[dict[str, Any]]:
    """es-ES test rows of MASSIVE 1.0, the tarball behind AmazonScience/massive@ff6bd8e."""
    tar = cache / "massive-1.0.tar.gz"
    if not tar.is_file():
        tar.write_bytes(urllib.request.urlopen(MASSIVE_TGZ, timeout=120).read())
    with tarfile.open(tar) as blob:
        raw = blob.extractfile("1.0/data/es-ES.jsonl").read()
    return [row for row in map(msgspec.json.decode, raw.splitlines()) if row["partition"] == "test"]


def rows_of(dataset: str, config: str, split: str, total: int) -> list[dict[str, Any]]:
    """Every row of a datasets-server slice, 100 per page."""
    out: list[dict[str, Any]] = []
    for offset in range(0, total, 100):
        url = ROWS_API.format(dataset=dataset, config=config, split=split, offset=offset, length=min(100, total - offset))
        page = msgspec.json.decode(urllib.request.urlopen(url, timeout=60).read())  # noqa: S310
        out.extend(row["row"] for row in page.get("rows", []))
    return out


def pick(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    picks = np.random.default_rng(SEED).choice(len(rows), size=min(size, len(rows)), replace=False)
    return [rows[int(index)] for index in picks]


def build_rows(size: int, cache: Path) -> list[dict]:
    full = massive_es(cache)
    intents = sorted({row["intent"] for row in full})
    massive = pick(full, size)
    sentim = pick(rows_of("cardiffnlp/tweet_sentiment_multilingual", "spanish", "test", 1000), size)
    xnli = pick(rows_of("facebook/xnli", "es", "validation", 1000), size)
    return [
        *(
            {
                "source": "massive_intent_es",
                "state": row["utt"],
                "label": row["intent"],
                "question": {
                    "type": "choice",
                    "instructions": "¿Qué intención expresa este mensaje?",
                    "criteria": dict.fromkeys(intents),
                },
            }
            for row in massive
        ),
        *(
            {
                "source": "massive_intent_es20",
                "state": row["utt"],
                "label": row["intent"],
                "question": {
                    "type": "choice",
                    "instructions": "¿Qué intención expresa este mensaje?",
                    "criteria": narrow(row["intent"], intents, index),
                },
            }
            for index, row in enumerate(massive)
        ),
        *(
            {
                "source": "tweets_sentiment_es",
                "state": row["text"],
                "label": str(row["label"]),
                "question": {"type": "score", "instructions": "¿Cuál es el sentimiento de este tweet?", "criteria": SENTIMENT},
            }
            for row in sentim
        ),
        *(
            {
                "source": "xnli_es",
                "state": row["premise"],
                "label": "true" if int(row["label"]) == 0 else "false",
                "question": {"type": "noul", "instructions": f"{TRUE_GIVEN}\n{row['hypothesis']}"},
            }
            for row in xnli
        ),
    ]


def narrow(gold: str, all_intents: list[str], index: int) -> dict[str, None]:
    """The 20-option question Laya evaluates: gold plus 19 distractors, order seeded per row."""
    others = [intent for intent in all_intents if intent != gold]
    rng = random.Random(f"{SEED}:{index}")  # noqa: S311 — deterministic sampling, not crypto
    keys = [gold, *rng.sample(others, INTENTS_N)]
    rng.shuffle(keys)
    return dict.fromkeys(keys)


def main() -> None:
    out, size = Path(sys.argv[1]), int(next(iter(sys.argv[2:3]), DEFAULT_SIZE))
    rows = build_rows(size, Path(".cache"))
    out.write_bytes(b"\n".join(msgspec.json.encode(row) for row in rows) + b"\n")
    print(f"{len(rows)} labeled examples -> {out}")


if __name__ == "__main__":
    main()
