"""scripts/backend_watch.py: sample peak VRAM and peak RSS while a backend answers.

    uv run python scripts/backend_watch.py OUT.json --seconds 600 [--pid PID ...] [--cgroup memory.peak ...] [--gpu]

VRAM is `nvidia-smi --query-compute-apps` per pass: the sum, and each pid's own peak — the difference from
the first pass is what the backend added to the card, the per-pid peak is its own. RSS is the kernel's high
water mark (`VmHWM` per process, `memory.peak` per cgroup), so peaks survive the process."""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

GPU_FIELDS = "pid,used_memory"


async def _gpu_apps() -> dict[int, int]:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi", f"--query-compute-apps={GPU_FIELDS}", "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    rows: dict[int, int] = {}
    for line in out.decode().splitlines():
        match re.match(r"^\s*(\d+)\s*,\s*(\d+)", line):
            case [_, pid, mib]:
                rows[int(pid)] = int(mib) * 1024 * 1024
    return rows


async def _vm_hwm(pid: int) -> int:
    status = await asyncio.to_thread(Path(f"/proc/{pid}/status").read_text)
    return next(_kb(line) for line in status.splitlines() if line.startswith("VmHWM:"))


def _kb(line: str) -> int:
    return int(line.split()[1]) * 1024


async def watch(out: Path, pids: list[int], cgroups: list[Path], gpu: bool, seconds: float, interval: float) -> None:
    peak_rss = 0
    per_pid: dict[int, int] = {}
    baseline = total = 0
    started = time.perf_counter()
    first = True
    while time.perf_counter() - started < seconds:
        peak_rss = sum(await _vm_hwm_safe(pid) for pid in pids)
        for cgroup in cgroups:
            match await _read_cgroup(cgroup):
                case value:
                    peak_rss = max(peak_rss, value)
        match gpu:
            case True:
                apps = await _gpu_apps()
                for pid, used in apps.items():
                    per_pid[pid] = max(per_pid.get(pid, 0), used)
                total = max(total, sum(apps.values()))
                baseline = sum(apps.values()) if first else baseline
        first = False
        await asyncio.sleep(interval)
    out.write_text(
        json.dumps(
            {
                "peak_rss_bytes": peak_rss,
                "peak_gpu_bytes": total if gpu else 0,
                "baseline_gpu_bytes": baseline,
                "backend_gpu_bytes": max(0, (total - baseline)) if gpu else 0,
                "per_pid_gpu_bytes": per_pid,
                "seconds": round(time.perf_counter() - started, 1),
            }
        )
    )


async def _vm_hwm_safe(pid: int) -> int:
    try:
        return await _vm_hwm(pid)
    except (OSError, StopIteration):
        return 0


def _as_path(text: str) -> Path:
    return Path(text)


async def _read_cgroup(path: Path) -> int:
    try:
        raw = (await asyncio.to_thread(path.read_text)).strip()
    except OSError:
        return 0
    return int(raw) if raw.isdigit() else 0


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pid", action="append", type=int, default=[])
    parser.add_argument("--cgroup", action="append", type=_as_path, default=[])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--interval", type=float, default=0.2)
    args = parser.parse_args()
    asyncio.run(watch(args.out, args.pid, args.cgroup, args.gpu, args.seconds, args.interval))
    print(f"watched -> {args.out}")


if __name__ == "__main__":
    main()
