# TODO — vLLM changed under jev (2026-10-14)

The local vLLM (`:45100`, `vllm.service`) no longer serves Qwen3.6-27B. Read this before the next bench.

## What changed

- Model: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, served as **`gemma4-26b-a4b`** (MoE, 4B active). The old weights are
  deleted from the HF cache.
- `qwen3.6-27b` is still accepted as a **temporary alias of Gemma** so nothing breaks; it will be removed. A run that
  asks for `qwen3.6-27b` today is measuring Gemma.
- Image pinned `vllm/vllm-openai:v0.31.0` (the unit used to pull `latest` on every start).
- `--scheduling-policy priority`: lower value is served first; requests without `priority` are 0.
- 16 seqs, 32k context, CUDA graphs, fp8 KV (143k tokens), thinking off by default (`enable_thinking` still honoured).
- Throughput: 722 tok/s at concurrency 16 (Qwen 27B dense: ~135 at 8). Every earlier jev number is not comparable.

## The GPU budget jev lives in (RTX 4090, 24 GiB)

| tenant | VRAM | note |
|---|---|---|
| desktop (Xorg, shell, Zed, Electron) | ~1.4 GiB, must keep ≥ 1.8 GiB free | below ~0.5 GiB free the desktop freezes |
| vLLM `gemma4-26b-a4b` (`vllm.service`) | ~19.3 GiB fixed (`--gpu-memory-utilization 0.83`) | weights 15.3 + fp8 KV ~90k tokens + graphs |
| TEI `embed` Qwen3-Embedding-0.6B (`embed.service`) | ~1.7 GiB idle, ~2.2 GiB peak | fp16, 8192 batch tokens; CortexDB's embeddings |

- **The GPU is full.** No other GPU container (llama.cpp, a second vLLM, a GPU jev backend) may start without
  re-budgeting `VLLM_GPU_UTIL` first: a 7 GB llama.cpp started beside vLLM on 2026-10-14 put it in a restart loop.
- vLLM budgets over *total* VRAM: at startup it needs `0.83 × 23.5 GiB` free or it refuses to boot.
- Both units run `docker compose pull` on start and `down` on stop: a stopped container disappears; images are pinned.
- **Embedding bursts share the GPU with jev.** TEI and vLLM time-slice the card; under continuous embedding vLLM
  throughput halves (871 → 423 tok/s at concurrency 16). A normal CortexDB ingest is < 1 s per document, but a
  bulk re-index during a bench will skew its numbers — check `docker logs embed` for activity before trusting a run.
- jev is bulk load: it shares the 16 seats with CortexDB (extraction, enrichment, verifier), OpenHuman
  (memory/learn roles) and openwebui. Priority and the concurrency cap below are what keep them alive.

## What jev must do

1. **Model name.** `src/e_jev/config/vllm.py` `model = "qwen3.6-27b"` → `"gemma4-26b-a4b"`; same in `README.md:16`
   and the compose default (`JEV_MODEL=${VLLM_SERVED_NAME:-…}` in `~/.config/compose/jev/compose.yml` and the
   e-templates `stack/llm/jev` template, already switched on `feat/darts-engine`).
2. **Priority.** jev is bulk load and starved CortexDB before. Send `"priority": 10` in every request's `extra_body`
   (`adapters/reader/vllm.py` — the readout `completions.create` at ~175 and the extraction call at ~194), so memory
   and interactive clients (priority 0) are served first.
3. **Concurrency.** `config/systemone.py` `concurrency = 16` equals `--max-num-seqs 16`: jev alone fills every seat.
   Cap it at ≤ 8 so the other consumers always find a free slot even before priority preemption kicks in.
4. **Tokenizer-dependent startup — re-verify.** Gemma's tokenizer and chat template differ from Qwen's:
   - `_open_labels`: every label in `UNIVERSE` is re-tokenized; check the readout still treats multi-token labels
     correctly (trie depth may change).
   - `_open_stop`: the end-of-turn probe (`PROBE = "Z"`) must still find the stop token in Gemma's template — assert
     it at startup rather than trusting it.
   - `NO_THINK` stays valid (Gemma's template reads `enable_thinking`).
   - Outlines extraction → structured outputs (xgrammar) works on Gemma; re-run the extraction tests.
5. **Re-baseline.** Re-run calibration and the reference benches on Gemma; tag old results as `qwen3.6-27b` so they are
   not mixed with new ones.
6. **Tell the vLLM owner** when jev no longer sends `qwen3.6-27b`, so the alias can be dropped.

## Where the evidence is

- A/B harness and raw results: `/tmp/llm-ab/` (`bench.py`, `hard.py`, `results/*.json`) — temporary.
- Live config: `~/.config/compose/vllm/compose.yml` (comments carry the numbers).
- PRs: e-templates #9 (vllm + cortexdb templates), e-engineer #37 (memory/learn roles).
