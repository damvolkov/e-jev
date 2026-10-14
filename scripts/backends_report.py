"""scripts/backends_report.py: one table and a Pareto plot from the bake-off JSONs, into docs/evals.md's shape.

    uv run --with matplotlib python scripts/backends_report.py .reports/<DATE>_backends

Reads every <CONFIG>.json produced by backend_bench plus its watch_<CONFIG>.json peaks and the
<CONFIG>_evals.json scores, prints the config-by-metric table and writes docs/evals/backends.svg:
peak VRAM (or RSS when there is no VRAM) against the WorkflowEvals mean — the memory-efficiency frontier
the plan governs by."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("peak_vram_gb", "peak_rss_gb", "p95_noul_s", "throughput_c32", "ece_after_max", "accuracy_min", "evals_mean", "cold_s")


def load(directory: Path) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    for file in sorted(directory.glob("*.json")):
        config, data = config_of(file.name), json.loads(file.read_text())
        row = rows.setdefault(config, dict.fromkeys(METRICS))
        match file.name:
            case name if name.startswith("watch_"):
                row["peak_vram_gb"] = round((data.get("backend_gpu_bytes") or 0) / 2**30, 2) or round(
                    (data.get("peak_gpu_bytes") or 0) / 2**30, 2
                )
                row["peak_rss_gb"] = round((data.get("peak_rss_bytes") or 0) / 2**30, 2) or None
            case name if name.endswith("_evals.json"):
                row["evals_mean"] = data.get("mean")
            case name if name.endswith(".json"):
                fit = data.get("fit", {})
                latency = data.get("latency", {}).get("c32", {})
                row["throughput_c32"] = round(latency.get("throughput", 0), 2) or None
                row["p95_noul_s"] = percentile_of(latency, "boolq", 95)
                row["ece_after_max"] = max_or_none(fit.get("ece_after", {}).values())
                row["accuracy_min"] = min_or_none(fit.get("accuracy", {}).values())
    return rows


def config_of(name: str) -> str:
    return name.removeprefix("watch_").removesuffix("_evals.json").removesuffix(".json")


def percentile_of(latency: dict, source: str, pct: int) -> float | None:
    per_source = latency.get("per_source", {}).get(source)
    match per_source:
        case None:
            return None
        case values:
            return round(values.get(f"p{pct}", 0), 2) or None


def max_or_none(values) -> float | None:
    present = [round(v, 3) for v in values if v is not None]
    return max(present) if present else None


def min_or_none(values) -> float | None:
    present = [round(v, 3) for v in values if v is not None]
    return min(present) if present else None


def table(rows: dict[str, dict[str, float | None]]) -> str:
    header = "| config | " + " | ".join(METRICS) + " |"
    rule = "|" + "---|" * (len(METRICS) + 1)
    body = "\n".join(
        f"| {config} | " + " | ".join("" if row[m] is None else str(row[m]) for m in METRICS) + " |" for config, row in sorted(rows.items())
    )
    return "\n".join((header, rule, body))


def pareto(rows: dict[str, dict[str, float | None]], out: Path) -> None:
    points = [
        (r["peak_vram_gb"] or r["peak_rss_gb"] or 0, r["evals_mean"] or 0, name) for name, r in rows.items() if r["evals_mean"] is not None
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for memory, quality, name in points:
        ax.scatter(memory, quality, s=60)
        ax.annotate(name, (memory, quality), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("peak memory, GB (VRAM where it exists, else RSS)")
    ax.set_ylabel("WorkflowEvals mean agreement")
    ax.set_title("decision backends — memory vs quality")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main() -> None:
    directory = Path(sys.argv[1])
    rows = load(directory)
    print(table(rows))
    for file in sorted(directory.glob("*.json")):
        if file.name.startswith("watch_"):
            print(file.name, json.loads(file.read_text()))
    pareto(rows, Path("docs/evals/backends.svg"))


if __name__ == "__main__":
    main()
