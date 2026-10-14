"""scripts/backends_report.py: one config-by-metric table and the memory/quality Pareto from the bake-off JSONs.

    uv run --with matplotlib python scripts/backends_report.py .reports/<DATE>_backends

Reads every <CONFIG>.json from backend_bench plus its watch_<CONFIG>.json peaks and <CONFIG>_evals.json
scores. Prints the table (accuracy and ECE on the held-out halves, p95 at c1/c32, decisions/s at c32,
peak VRAM/RSS) and writes docs/evals/backends.svg: memory against WorkflowEvals mean."""

import json
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("evals_mean", "acc_min", "ece_worst", "p95_c1", "p95_c32", "dec_s_c32", "vram_gb", "rss_gb")


def _max(values: Iterable[float]) -> float | None:
    present = [round(value, 3) for value in values if value is not None]
    return max(present) if present else None


def _min(values: Iterable[float]) -> float | None:
    present = [round(value, 3) for value in values if value is not None]
    return min(present) if present else None


def load(directory: Path) -> dict[str, dict[str, float | None]]:
    rows: dict[str, dict[str, float | None]] = {}
    for file in sorted(directory.glob("*.json")):
        data = json.loads(file.read_text())
        if file.name.startswith("watch_"):
            config = file.name.removeprefix("watch_").removesuffix(".json")
            vram = (max((data.get("per_pid_gpu_bytes") or {0: 0}).values(), default=0)) / 2**30
            row = rows.setdefault(config, dict.fromkeys(METRICS))
            row["vram_gb"] = round(vram, 2) or row["vram_gb"]
            row["rss_gb"] = round((data.get("peak_rss_bytes") or 0) / 2**30, 2) or row["rss_gb"]
            continue
        if "fit" not in data:
            config = file.name.removesuffix("_evals.json").removesuffix(".json")
            rows.setdefault(config, dict.fromkeys(METRICS))["evals_mean"] = data.get("mean")
            continue
        config = data["config"]
        row = rows.setdefault(config, dict.fromkeys(METRICS))
        latency = data.get("latency", {})
        row["acc_min"] = _min(data["fit"]["accuracy"].values())
        row["ece_worst"] = _max(data["fit"]["ece_after"].values())
        row["p95_c1"] = latency.get("c1", {}).get("p95")
        row["p95_c32"] = latency.get("c32", {}).get("p95")
        row["dec_s_c32"] = round(latency.get("c32", {}).get("throughput", 0), 1) or None
    return rows


def table(rows: dict[str, dict[str, float | None]]) -> str:
    header = "| config | " + " | ".join(METRICS) + " |"
    rule = "|" + "---|" * (len(METRICS) + 1)
    body = "\n".join(
        f"| {config} | " + " | ".join("—" if row[m] is None else str(row[m]) for m in METRICS) + " |"
        for config, row in sorted(rows.items())
    )
    return "\n".join((header, rule, body))


def pareto(rows: dict[str, dict[str, float | None]], out: Path) -> None:
    points = [(max(r["vram_gb"] or 0, r["rss_gb"] or 0), r["evals_mean"], name) for name, r in rows.items() if r["evals_mean"] is not None]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for memory, quality, name in points:
        ax.scatter(memory, quality, s=60)
        ax.annotate(name, (memory, quality), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("symlog", linthresh=0.2)
    ax.set_xlabel("peak memory GB (max of VRAM, RSS) — 0 = CPU-only")
    ax.set_ylabel("WorkflowEvals mean agreement")
    ax.set_title("decision backends — memory vs quality")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main() -> None:
    rows = load(Path(sys.argv[1]))
    print(table(rows))
    pareto(rows, Path("docs/evals/backends.svg"))
    print("wrote docs/evals/backends.svg")


if __name__ == "__main__":
    main()
