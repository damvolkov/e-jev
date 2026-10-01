"""scripts/calibration_build.py: a labeled calibration set from public datasets, one System One kind each.

    uv run --with datasets python scripts/calibration_build.py OUT.jsonl [PER_SOURCE]

BoolQ → noul · AG News → choice (4) · Banking77 → choice (77, past the letters) · SST-5 → score (5 levels).
"""

import sys
from pathlib import Path

import msgspec
import numpy as np
from datasets import load_dataset

SEED = 7
DEFAULT_SIZE = 300
SENTIMENT = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
NEWS = {"World": "World politics and events", "Sports": "Sport", "Business": "Business and economy", "Sci/Tech": "Science and technology"}


def sample_rows(name: str, split: str, size: int, config: str | None = None) -> list[dict]:
    rows = load_dataset(name, config, split=split)
    picks = np.random.default_rng(SEED).choice(len(rows), size=min(size, len(rows)), replace=False)
    return [rows[int(index)] for index in picks]


def build_rows(size: int) -> list[dict]:
    intents = load_dataset("legacy-datasets/banking77", split="test").features["label"].names
    return [
        *(
            {
                "source": "boolq",
                "state": row["passage"],
                "label": str(row["answer"]).lower(),
                "question": {"type": "noul", "instructions": row["question"].capitalize() + "?"},
            }
            for row in sample_rows("google/boolq", "validation", size)
        ),
        *(
            {
                "source": "ag_news",
                "state": row["text"],
                "label": list(NEWS)[row["label"]],
                "question": {"type": "choice", "instructions": "What is this news article about?", "criteria": NEWS},
            }
            for row in sample_rows("fancyzhx/ag_news", "test", size)
        ),
        *(
            {
                "source": "banking77",
                "state": row["text"],
                "label": intents[row["label"]],
                "question": {
                    "type": "choice",
                    "instructions": "Which intent does this bank customer message express?",
                    "criteria": dict.fromkeys(intents),
                },
            }
            for row in sample_rows("legacy-datasets/banking77", "test", size)
        ),
        *(
            {
                "source": "sst5",
                "state": row["text"],
                "label": str(row["label"]),
                "question": {"type": "score", "instructions": "What is the sentiment of this movie review?", "criteria": SENTIMENT},
            }
            for row in sample_rows("SetFit/sst5", "test", size)
        ),
    ]


def main() -> None:
    out, size = Path(sys.argv[1]), int(next(iter(sys.argv[2:3]), DEFAULT_SIZE))
    rows = build_rows(size)
    out.write_bytes(b"\n".join(msgspec.json.encode(row) for row in rows) + b"\n")
    print(f"{len(rows)} labeled examples -> {out}")


if __name__ == "__main__":
    main()
