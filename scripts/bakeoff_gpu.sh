# scripts/bakeoff_gpu.sh: the GPU bake-off in one vLLM-stopped window, every candidate measured alone on the card.
#
#   systemd-run --user --unit=bakeoff-gpu -p StandardOutput=append:/tmp/opencode/jev-bakeoff/gpu.log bash scripts/bakeoff_gpu.sh
#
# For each candidate: start → cold start (process to first answer) → watch peaks → bench → stop.
# vLLM and jev are stopped at entry and restored at exit on every path. Stop/start windows go to the report.
set -uo pipefail

REPO=/home/damien/workspaces/damien/e-jev
DATA=$HOME/.config/compose/data
R=$REPO/.reports/2026-10-13_backends
W=$R/windows.md
LAYA=~/.venvs/laya/bin/laya-serve
mkdir -p "$R"

stamp() { printf '%s %s\n' "$(date +%FT%T%z)" "$*" | tee -a "$W"; }

restore() {
  stamp "start vllm jev"
  systemctl --user start vllm jev
}
trap restore EXIT

cd "$REPO"
stamp "stop vllm jev (GPU window)"
systemctl --user stop jev vllm

# C1 — Clef-Flash Q4_K_M on llama.cpp, GPU.
docker run -d --name bake-clef --gpus all -p 45164:8080 -v "$DATA/clef/models:/models:ro" \
  ghcr.io/ggml-org/llama.cpp:server-cuda /app/llama-server -m /models/Clef-Flash-Q4_K_M.gguf --host 0.0.0.0 --port 8080 -c 32768 -ngl 99
CGROUP=$(docker inspect --format '{{.Id}}' bake-clef)
stamp "cold-start begin C1_clef_q4"
until curl -sf http://localhost:45164/health >/dev/null; do sleep 1; done
curl -s http://localhost:45164/health >/dev/null && stamp "cold-start end C1_clef_q4 first answer ready"
uv run python scripts/backend_watch.py "$R/watch_C1_clef_q4.json" --gpu \
  --cgroup "/sys/fs/cgroup/system.slice/docker-${CGROUP}.scope/memory.peak" --seconds 1500 &
WATCH_C1=$!
uv run python scripts/backend_bench.py C1_clef_q4 http://localhost:45164 --file .cache/labeled.jsonl --file .cache/spanish.jsonl --out "$R/C1_clef_q4.json" --lat-per-kind 30
stamp "bench end C1_clef_q4"
kill $WATCH_C1 2>/dev/null; docker rm -f bake-clef >/dev/null

# E1 — Kev-0.8b on its own server, GPU.
(cd /tmp/opencode/jev-bakeoff/kev && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 45165 --device cuda >/dev/null 2>&1 &)
stamp "cold-start begin E1_kev08"
until curl -sf "http://localhost:45165/v1/systemone" -X POST -H 'content-type: application/json' -d '{"model":"kev-0.8b","state":"x","questions":{"q":{"type":"noul","instructions":"t?"}}}' >/dev/null; do
  sleep 2
done
stamp "cold-start end E1_kev08 first answer ready"
uv run python scripts/backend_bench.py E1_kev08 http://localhost:45165 --file .cache/labeled.jsonl --file .cache/spanish.jsonl --out "$R/E1_kev08.json" --lat-per-kind 30 --model kev-0.8b
stamp "bench end E1_kev08"
pkill -f "kev.serve"

# B-cuda — both Laya checkpoints on the card, model field selects the checkpoint.
LAYA_DEVICE=cuda LAYA_HOST=127.0.0.1 LAYA_PORT=45166 LAYA_MODELS=english,multilingual LAYA_JEV_STRICT=1 "$LAYA" >/dev/null 2>&1 &
LAYA_PID=$!
stamp "cold-start begin B_cuda"
until curl -sf http://localhost:45166/health >/dev/null; do sleep 1; done
stamp "cold-start end B_cuda health ready"
uv run python scripts/backend_watch.py "$R/watch_B_cuda.json" --pid $LAYA_PID --gpu --seconds 1500 &
WATCH_B=$!
uv run python scripts/backend_bench.py B1_laya_multi_cuda http://localhost:45166 --file .cache/labeled.jsonl --file .cache/spanish.jsonl --out "$R/B1_laya_multi_cuda.json" --model multilingual --lat-per-kind 30
uv run python scripts/backend_bench.py B2_laya_en_cuda http://localhost:45166 --file .cache/labeled.jsonl --file .cache/spanish.jsonl --out "$R/B2_laya_en_cuda.json" --model english --lat-per-kind 30
stamp "bench end B_cuda"
kill $WATCH_B $LAYA_PID 2>/dev/null; wait 2>/dev/null

# C2 — Clef-Flash Q8_0, GPU.
docker run -d --name bake-clef8 --gpus all -p 45168:8080 -v "$DATA/clef/models:/models:ro" \
  ghcr.io/ggml-org/llama.cpp:server-cuda /app/llama-server -m /models/Clef-Flash-Q8_0.gguf --host 0.0.0.0 --port 8080 -c 32768 -ngl 99
CG8=$(docker inspect --format '{{.Id}}' bake-clef8)
until curl -sf http://localhost:45168/health >/dev/null; do sleep 1; done
uv run python scripts/backend_watch.py "$R/watch_C2_clef_q8.json" --gpu --cgroup "/sys/fs/cgroup/system.slice/docker-${CG8}.scope/memory.peak" --seconds 1500 &
WATCH_C2=$!
uv run python scripts/backend_bench.py C2_clef_q8 http://localhost:45168 --file .cache/labeled.jsonl --file .cache/spanish.jsonl --out "$R/C2_clef_q8.json" --lat-per-kind 30
stamp "bench end C2_clef_q8"
kill $WATCH_C2 2>/dev/null; docker rm -f bake-clef8 >/dev/null

stamp "GPU window done — vLLM and jev start next on trap"
