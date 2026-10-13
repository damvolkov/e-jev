# NEWPLAN — e-jev 2: a real decision-model service

Hand-off for the next instance. Written 2026-10-13. Every external fact below was read from its primary source
on that date; anything marked **UNVERIFIED** must be checked before it is relied on.

## 1. Why

Today e-jev answers the TypeSafe System One contract by reading option-letter logprobs of a generative 27B
(`cyankiwi/Qwen3.6-27B-AWQ-INT4`, vLLM, 21.9 GB of the 4090). That is the SemIf method: honest, calibrated,
on par with Jev on WorkflowEvals — and 10–60× slower than Jev, while holding the whole GPU.

Since September a family of **trained decision models** exists: encoder or decoder backbones with decision
heads, one forward pass, no generation, several already serving `POST /v1/systemone` natively. e-jev 2 keeps
what e-jev already does well — the contract, the CLI, Phoenix traces, per-kind calibration, the two eval
harnesses — and puts those models behind it.

**Governing criterion: memory efficiency first.** Among configurations that are *usable*, pick the one with the
smallest resident footprint (VRAM first, then RAM). Quality only breaks ties beyond the usable bar, and the bar
is defined in §5 before any run, not after.

## 2. Authorization (granted by Damien, 2026-10-13)

- You may **stop vLLM** (`sudo systemctl stop vllm jev`) to free the 4090 for the bake-off, and run any
  candidate stack on it.
- Stopping vLLM takes down: e-jev (current backend), OpenHuman's local model on `:45100`
  (`~/.config/openhuman/users/*/config.toml`), Open WebUI, mem0. Batch the GPU runs; do not leave it stopped
  between sessions.
- **Restore when done or when stopping work**: `sudo systemctl start vllm jev`, then `make status` and
  `nvidia-smi` showing `VLLM::EngineCore`. Log every stop/start window in the run report.
- Do not touch `llamacpp` compose's default model; run candidates in their own containers/ports
  (45161–45169 reserved for this plan).
- Weights go under `~/.config/compose/data/<service>/models`; delete what loses the bake-off.

## 3. Candidates (evidence)

| Id | Model | Kind | Serve | Footprint (claimed) | Contract | Notes |
|---|---|---|---|---|---|---|
| A | current e-jev, Qwen3.6-27B AWQ | logprob reading | vLLM | 21.9 GB VRAM (measured) | native | baseline |
| B1 | **Laya-multilingual** (mmBERT-base 322M) | encoder + heads | `pip install "laya[serve]"` → `laya-serve`, `LAYA_DEVICE=cpu\|cuda` | <1 GB; CPU 48 ms/q on 28 threads | `/v1/systemone` with diffs (§6) | **Spanish**; ships *no* temperatures; degrades >4k tokens; weak wide choice (Banking77 0.425) |
| B2 | Laya (ModernBERT-large 421M) | encoder + heads | same | <1 GB | same | English; ECE 0.466 raw → 0.081 after temperature |
| C1 | **Clef-Flash** Q4_K_M | Qwen3.5-9B + joint schema head | llama.cpp ≥ PR #29831 (merged 2026-10-03), `ggml-org/Clef-Flash-GGUF` | ~5.5 GB (**UNVERIFIED**, arithmetic) | native on llama-server | Banking77 90.9 vs Jev 79.7; loses on GPQA/BBH/MMLU-Pro |
| C2 | Clef-Flash Q8_0 | same | llama.cpp, GPU or CPU | ~10 GB (**UNVERIFIED**) | native | CPU latency unknown |
| C3 | Clef-Flash BF16 | same | `mode-io/vllm-jev` (`vllm-jev serve Cloudflare/clef-flash`) | ~19 GB | native | quality ceiling of C |
| D | **jevos-v4** (feder-cr/jev, MIT) | MiniCPM cut to 17 layers + head | one C++ binary, port 8017, OpenVINO INT8 / GGUF | ~1–1.4 GB **RAM**, 28–130 ms on a laptop CPU | native | English only; below Jev on all its tasks |
| E1 | Kev-0.8b (Apache) | Qwen3.5 + LoRA + pointer head | `uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b` | 3.8 GB VRAM | native, SDK unchanged | English only; BoolQ/AG News/Banking77/SST-5 are **in its training set** → only WorkflowEvals counts |
| E2 | Kev-4b | same | same | 14.3 GB | native | same caveat |
| F | Ollama ≥ 0.35 `/v1/systemone` | whatever it serves | ollama | — | native (release notes) | **UNVERIFIED** which decision models it ships; check before including |

Rejected, do not spend time: pplx-decider (~49 GiB, own runtime), Foq (logprob reading, 11★, forked runtime),
SemIf (= A), NanoJev (games only, dormant), OpenJev (CC BY-NC).

Sources: github.com/NandhaKishorM/laya (README, BENCHMARKS.md) · huggingface.co/Cloudflare/clef-flash ·
huggingface.co/ggml-org/Clef-Flash-GGUF · github.com/ggml-org/llama.cpp/pull/29831 · github.com/mode-io/vllm-jev ·
github.com/feder-cr/jev (release jevos-v4) · github.com/jaredpalmer/kev (model cards) ·
github.com/AnotiaWang/awesome-decision-models · github.com/AbdelStark/awesome-typesafe-jev.

Also in the awesome list, unverified individually — screen with one `gh api` each, include only if a
self-hostable `/v1/systemone` server and ≤8 GB exist: Strands Decider 2B (AWS), Decision 2.0 (vllm-sr, 0.6B+),
JevK5 (2B GGUF), Standard One (3B GGUF), Phocinae-Largha-150M.

## 4. Bake-off

### Harnesses (already in the repo — reuse, do not rewrite)

1. **Calibration set** — `scripts/calibration_build.py`, `make calibrate`: BoolQ (noul), AG News (choice 4),
   Banking77 (choice 77), SST-5 (score 5); fit on one half, report on the held-out half.
2. **WorkflowEvals** — `make evals` with `--base-url` override, `EVALS_LIMIT=12` (48 cases), scored against
   Jev 1.13's published runs (`docs/evals.md`).
3. **Spanish set (new)** — MASSIVE `es-ES` intents (choice, the set Laya reports on) plus one Spanish noul and one
   score set; pick public ones, record the exact dataset id and revision. If none fits, build ≥200 labeled
   examples from Damien's own traffic (Phoenix traces) and say so.

Point both harnesses at a candidate by base URL only. A candidate that cannot be reached through the TypeSafe
contract goes through the gateway adapter (§6), never through a bespoke eval path.

### Measured per configuration

| Metric | How |
|---|---|
| peak VRAM | `nvidia-smi --query-compute-apps=pid,used_memory --format=csv -lms 200` during the run, max |
| peak RSS | `/proc/<pid>/status` VmHWM, or `docker stats --no-stream` sampled |
| accuracy | held-out half per dataset; WorkflowEvals agreement |
| ECE | before and after per-kind temperature (fit on the fit half) |
| latency | p50 / p95 per question kind, single request and at concurrency 16 and 32 (jevgrep uses 32) |
| throughput | decisions/s at concurrency 32 |
| cold start | process start → first answer |

Run every GPU configuration with vLLM stopped, alone on the card. Then the **co-residence runs** (§4.3).

### 4.1 Order (cheapest first, stop early when the frontier is clear)

1. CPU-only: B1, B2, D (zero VRAM — the efficiency ideal; if one clears the bar, it is the default backend).
2. Small GPU: B1/B2 on cuda, C1, E1.
3. Medium GPU: C2, E2.
4. Ceiling: C3 (only if C1/C2 fall short of A on quality and the gap must be priced).
5. Baseline A re-measured with the same harness (latency at concurrency 32 is new).

### 4.2 Decision rule (fixed before running)

- **Usable** = WorkflowEvals mean ≥ A's (0.573) − 0.05 **and** held-out accuracy ≥ A's − 5 pts on AG News and
  BoolQ **and** p95 < 1 s for one noul **and** post-calibration ECE ≤ 0.10.
- Among usable configurations, order by peak VRAM, then peak RSS, then p95. The first is the **default backend**.
- If none is usable on everything, assign by question shape (§6 dispatch) and require each route to be usable on
  its own slice (e.g. Banking77-like wide choice may route to C while short noul goes to B/D).
- Spanish: the default for Spanish states must be usable on the Spanish set; English-only models never get it.

### 4.3 Co-residence

The end state must leave room for the 27B unless the bake-off proves it unnecessary. For the winner(s), measure
with vLLM restarted at a lower `--gpu-memory-utilization` (try 0.80, 0.70, 0.65) and its current
`--max-model-len 20480`: does vLLM still start, what KV capacity remains (vLLM logs "GPU KV cache size"), and
does OpenHuman chat on `:45100` still work. CPU winners need no co-residence run.

### Output

`docs/evals.md` gains a "Decision backends" section: one table (config × metric), the Pareto plot memory vs
WorkflowEvals mean (`docs/evals/backends.svg`), and the decision with its numbers. Raw results under
`.reports/<date>_backends/` (JSON per run). Include the stop/start windows of vLLM.

## 5. Architecture of e-jev 2

Hexagonal, as today. The current port `adapters/ports.py::Reader` is *logprob-level*; decision models answer
*whole requests*. Add one port at the request level and keep `Reader` as one adapter behind it.

```
api (/v1/systemone, /v1/models, /v1/extract)
  └── operational/engine  →  Decider port  (decide(Request) -> Response)
        ├── adapters/decider/reading   — today's Reader + vLLM path (teacher, fallback)
        ├── adapters/decider/systemone — any native /v1/systemone upstream (Clef via llama.cpp, jevos, Kev, vllm-jev)
        └── adapters/decider/laya      — laya-serve, with the contract fixes of §6
```

- **Dispatch**: a frozen `Mapping[Route, Decider]` chosen per *question* by shape — language (Spanish → multilingual
  route), state tokens (> 4k → not Laya), option count (> 20 → not Laya; > 255 rejected), kind. Requests are split
  per route and merged; questions in one request are already independent in the contract.
- **Calibration** stays e-jev's: one temperature per (backend, kind), fitted with `make calibrate` per backend.
  Thresholds tuned on one backend do not transfer (documented in `docs/evals.md`).
- **Fallback**: upstream error or timeout → next route in the row; never silently answer from a weaker route
  without marking it in the trace.
- **Traces**: one Phoenix span per route call with backend id, peak tokens, latency.
- **Settings**: `JEV_BACKENDS__*` in pydantic-settings; no `os.environ`.

### Contract fixes (independent of the bake-off, do first)

- Return **429/529** on overload instead of 502 (the official SDKs retry 429/529; 502 retry **UNVERIFIED**).
- Raise context to **32k** where the backend allows (TypeSafe's limit; e-jev is at 20k).
- Laya adapter: recompute `confidence` with Jev's formula `(p_max − 1/K)/(1 − 1/K)` from `probabilities`
  (Laya returns 1 − normalised entropy); map `null` score levels to their index text or 422; route > 20 options to
  Laya's `predict_tournament` or away from Laya.

## 6. Distillation loop (after the default backend is chosen)

The slow path teaches the fast path. Reuse before writing:

- **stuntd** (github.com/bladedevoff/stuntd, 70★, Apache-2.0, pushed 2026-10-11): local proxy that learns an app's
  typed decisions and answers them with a Laya head; Jev- and OpenAI-compatible.
- **Jevstiller** (awesome-typesafe-jev): proxy behind `TYPESAFE_BASE_URL` distilling a local head from Jev answers.

Plan: log (state, question, teacher probabilities) from route A to a JSONL sink; periodically train per-decision
Laya heads (stuntd recipe) for the highest-volume question signatures; promote a head only when it beats its
route's held-out accuracy and ECE.

## 7. Consumers to validate at the end

| Consumer | Hook | Check |
|---|---|---|
| Pydantic AI `TypeSafeModel` | `examples/triage/graph.py` | `make integration` green on the new default |
| n8n TypeSafe node | `deploy/n8n` | demo workflow runs |
| **jevgrep** (`jg`) | `printf local \| jg auth --provider custom --base-url http://localhost:45160/v1 --model jev-latest --stdin && jg doctor` | real searches on e-engineer finish under its **15 s** timeout at concurrency 32; noul only; states carry full source snippets → long-context route |
| **pg-jev** | `SET jev.api_url='http://127.0.0.1:45160/v1/systemone'` | needs `plpython3u` + superuser; raise `jev.timeout` for 20-row batches |
| **MCP server** (new) | expose `decide` (noul/choice/score) as an MCP tool from e-jev | none exists in the ecosystem; registers in e-engineer `core/*/config/mcp.yaml` for Claude Code, OpenCode and OpenHuman |

## 8. Work rules

- One branch for this thread: `work/decision-backends`, one PR. TDD on the new port and dispatch; `make check`
  green before every commit.
- Order: §5 contract fixes → §4 bake-off (report committed) → §5 Decider port + winning adapters → §7 consumers →
  §6 distillation.
- Stop and report instead of improvising if: no candidate is usable (§4.2), the 27B no longer fits after co-residence,
  or a key source fact here turns out false.
