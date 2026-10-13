# scripts/evals_bakeoff.sh: WorkflowEvals against one endpoint, scored like `make evals` under its own name.
#
#   scripts/evals_bakeoff.sh CONFIG BASE_URL [LIMIT]
#
# Reuses the pinned WorkflowEvals checkout and its own scoring: runs every workflow, then computes the
# agreement with the consensus reference on the shared cases, exactly as scripts/evals_compare.py does for
# e-jev. Writes .reports/<DATE>_backends/<CONFIG>_evals.json with the per-workflow and mean numbers.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
REV=0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb
OUT="$REPO/.reports/$(date +%F)_backends"

[ $# -ge 2 ] || { echo "usage: $0 CONFIG BASE_URL [LIMIT]"; exit 2; }
CONFIG=$1 BASE=$2 LIMIT=${3:-12}
mkdir -p "$OUT"

DIR="$REPO/.cache/workflowevals"
[ -d "$DIR" ] || git clone -q https://github.com/typesafe-ai/WorkflowEvals "$DIR"
git -C "$DIR" checkout -q "$REV"
cd "$DIR" && uv sync --locked -q

for w in invoice_processing customer_service agent_trace_observability security_incidents; do
  TYPESAFE_API_KEY=local uv run python run.py "$w" --base-url "$BASE" --name "$CONFIG" --limit "$LIMIT" --workers 2 >/dev/null 2>&1 || echo "$w failed"
done

uv run --extra plot python - "$CONFIG" "$OUT/${CONFIG}_evals.json" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from plot import combine, published_runs, workflow_points

config, out = sys.argv[1], sys.argv[2]
scores = {}
for workflow, metric in (
    ("invoice_processing", "exact"),
    ("customer_service", "exact"),
    ("agent_trace_observability", "primary"),
    ("security_incidents", "exact"),
):
    bundle = combine([Path(f"runs/{workflow}/{config}")], "consensus", published_runs(workflow))
    points = workflow_points(bundle, metric, "time", None)
    ours = next((p.y for p in points if p.model.name == config), None)
    scores[workflow] = ours
scores["mean"] = sum(v for v in scores.values() if v is not None) / max(1, sum(1 for v in scores.values() if v is not None))
Path(out).write_text(json.dumps(scores, indent=1))
print(json.dumps(scores))
PY
