"""scripts/evals_compare.py: e-jev's WorkflowEvals runs scored against the published ones, on shared cases.

Run from the pinned WorkflowEvals checkout (make evals does): uses its own loading and scoring code.
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")
from plot import combine, published_runs, workflow_points

for workflow, metric in (
    ("invoice_processing", "exact"),
    ("customer_service", "exact"),
    ("agent_trace_observability", "primary"),
    ("security_incidents", "exact"),
):
    bundle = combine([Path(f"runs/{workflow}/e-jev")], "consensus", published_runs(workflow))
    points = sorted(workflow_points(bundle, metric, "time", None), key=lambda p: -(p.y or 0))
    print(f"\n{workflow} · {metric} agreement with consensus (shared cases)")
    for p in points:
        print(f"  {p.model.name:<28} {p.y if p.y is None else round(p.y, 3):>6}   {'' if p.x is None else f'{p.x:.2f}s/case'}")
