<p align="center">
  <img src="assets/logo.svg" alt="e-jev" width="220" />
</p>

# e-jev

A self-hosted [Jev](https://docs.typesafe.ai): TypeSafe's System One API — typed, calibrated decisions
instead of text. Two decision paths behind one contract, sized to your GPU:

- **Clef-Flash** (Cloudflare, Apache-2.0): a trained decision model — one forward, every option scored
  jointly. 6.9 GB, noul p50 38 ms, 20.6 decisions/s, and it *beats* the generative path on quality
  (WorkflowEvals 0.639 vs 0.573, Banking77 0.961, Spanish intents 0.93–0.95).
- **The reader**: exact option-letter logprobs of a generative model (a Qwen served by vLLM),
  renormalized over your options — the teacher: calibrated, long-context, and the only extractor.

Wire-compatible with the official `typesafe-sdk`, `@typesafe-ai/sdk`, the TypeSafe n8n node, jevgrep and
pg-jev: point their base URL here. No text is generated on the decision path; confidence follows
TypeSafe's reference definitions; every answer is calibrated per (backend, question kind).

## Install

    docker pull ghcr.io/damvolkov/e-jev:latest
    docker run -d --name jev -p 45160:45160 --network e-core ghcr.io/damvolkov/e-jev:latest

With Clef on the same machine (the default decision backend — its model downloads on first run):

    docker run -d --name clef --gpus all -p 45164:8080 \
      -v ./clef-cache:/home/app/.cache/llama.cpp --network e-core \
      --entrypoint /bin/sh ghcr.io/ggml-org/llama.cpp:server-cuda -c \
      'exec /app/llama-server -hf ggml-org/Clef-Flash-GGUF:Q4_K_M --host 0.0.0.0 --port 8080 -c 32768 -ngl 99 -b 32768 -ub 4096'

and tell jev the fleet: `JEV_BACKENDS='[{"id":"clef","kind":"systemone","url":"http://clef:8080"}]'`
(`JEV_ROWS` optionally orders routes per question kind; the first reachable route that fits the shape
answers, failures fall down the row).

For this machine, `make deploy` installs everything as systemd --user units under `~/.config/compose`:

| Service | Port | Role |
|---|---|---|
| jev | 45160 | System One API — routing, calibration, traces |
| clef | 45164 | Clef-Flash Q4_K_M (llama.cpp, GGUF self-downloads) |
| vllm | 45100 | Qwen3.6-27B AWQ — teacher: `extract`, states > ~4k tokens, distillation |
| n8n / phoenix | 45700 / 45900 | optional UI and traces |

One 24 GB card holds clef (6.9) **or** the 27B teacher (21.6), not both:

    make gpu-decide   # clef + jev — the serving default
    make gpu-teach    # vllm + jev — extraction, long context, refitting

A backend that is down degrades to a failing route: the fleet boots and serves what it can; `/ready`
reports the survivors. `JEV_DECISION_LOG` records every answered question (state, scores, backend) as
JSONL — the corpus the distillation loop trains faster heads from.

## API

`POST /v1/systemone` (bare `/systemone` too — open clients join without the prefix) — the TypeSafe
contract: `state` (string, object or array), `model`, and a map of questions of type `noul`, `choice`
(≤255 options) or `score` (≤10 levels).

    curl -s localhost:45160/v1/systemone -H 'content-type: application/json' -d '{
      "state": "Help! My payouts have been failing for 3 days.",
      "model": "jev-latest",
      "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages", "sales": "Pricing"}},
        "is_urgent":  {"type": "noul", "instructions": "Does this convey urgency?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?", "criteria": ["Calm", "Frustrated", "Very angry"]}
      }}'

`GET /v1/models` — the fleet's names. `POST /v1/extract` — beyond Jev: typed values through the teacher.
Capacity answers 429/529 (what the official SDKs retry), never 502.

    from typesafe_sdk import Choice, TypeSafeClient
    client = TypeSafeClient(api_key="local", base_url="http://localhost:45160")

`ejev-mcp` (also `src/e_jev/mcp.py`) mounts the same decisions as MCP tools — noul, choice, score,
models — for Claude Code, OpenCode and OpenHuman over stdio.

## CLI

`ejev` asks any System One endpoint — this e-jev or TypeSafe's Jev — from the terminal:

    ejev noul "Is this urgent?" --state "Checkout is down, we lose orders"
    ejev choice "Which team?" "billing=Charges, refunds" "technical=Bugs" sales --state @ticket.txt
    ejev --json ask request.json
    ejev calibrate .cache/labeled.jsonl --backend clef   # per-backend fit, merged into the file

## Consumers verified

`examples/triage/graph.py` (Pydantic AI + pydantic-graph) — `make integration` and `make graph`;
n8n's official TypeSafe node — `make ui`; **jevgrep**: `jg auth --provider custom --base-url
http://localhost:45160/v1 --model jev-latest`, semantic search in ~2 s at its own c32; **pg-jev**:
`SET jev.api_url='http://jev:45160/v1/systemone'` inside Postgres (the setting takes the full endpoint) (plpython3u).

## Calibrate

One temperature per **(backend, kind)** — thresholds do not transfer across backends. `make calibrate`
fits the teacher, `ejev calibrate --backend clef` the decision model; the default set is BoolQ, AG News,
Banking77 (77-way), SST-5 plus a Spanish set (MASSIVE es-ES, XNLI es, tweet sentiment), each example in a
stable half: fit or held out. Every number reported comes from the held-out half. The bake-off table —
all measured candidates, quality and memory — lives in [docs/evals.md](docs/evals.md).

## Tuning

je v: `JEV_BACKENDS` / `JEV_ROWS` (JSON: the fleet and its preference rows), `JEV_PERMUTATIONS` (1; 2
averages both option orders on the reader), `JEV_CONCURRENCY` (16), `JEV_API_KEY`, `JEV_CALIBRATION`,
`JEV_DECISION_LOG`, `JEV_OTLP_ENDPOINT`. clef: `CLEF_CTX` (32768), `CLEF_UBATCH` (4096 — the decision
head needs the whole prompt in one batch), `CLEF_BATCH`. vllm: `VLLM_MODEL`, `VLLM_MAX_MODEL_LEN`
(32768 = TypeSafe's contract ceiling), `VLLM_GPU_UTIL` (0.93).

## Benchmark

Against TypeSafe's published runs on the same cases: the decision model is not merely cheaper than the
generative reader — it is **better** (0.639 vs 0.573 vs Jev's 0.583), at ⅓ the VRAM and 5× the speed.
Reproduce with `make evals`; the full bake-off: `docs/evals.md` and `.reports/`.

## Differences from Jev

- Long states (> ~4k tokens at default `CLEF_UBATCH`) route to the reader, which pays the generative
  model's latency; Jev's own budget is not published.
- `usage.output_tokens` counts forward passes on the reader, server-reported usage elsewhere.
- Probabilities come from the trained decision model or a general instruct model; calibrate per backend
  before trusting thresholds tuned on Jev's.
