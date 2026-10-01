# WorkflowEvals: e-jev against Jev

TypeSafe's own benchmark ([WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals), pinned at
`0ac3b8a`), run against e-jev through its `--base-url` override and scored with its own code against the
runs TypeSafe publishes on Hugging Face — Jev 1.13 and eight LLMs — on the same cases.

- **Date** 2026-10-01 · **e-jev** `cyankiwi/Qwen3.6-27B-AWQ-INT4` on one RTX 4090, vLLM 0.30.0, uncalibrated.
- **Sample** the first 12 cases of each workflow (48 of 705). Agreement with the consensus reference label.
- **Reproduce** `make evals` (`EVALS_LIMIT=12` by default).

| Workflow | Metric | e-jev | Jev 1.13 | best LLM | e-jev s/case | Jev s/case |
|---|---|---|---|---|---|---|
| invoice_processing | exact actions | **0.750** | 0.667 | 0.917 | 48.6 | 0.53 |
| customer_service | exact actions | 0.889 | **0.917** | 0.917 | 3.3 | 0.24 |
| agent_trace_observability | primary action | 0.458 | **0.583** | 0.917 | 8.3 | 0.51 |
| security_incidents | exact actions | **0.333** | 0.167 | 0.833 | 5.4 | 0.60 |
| mean | | 0.608 | 0.583 | | | |

## Reading it

- **Quality: on par with Jev on this sample.** Each workflow differs by one or two cases out of twelve, in
  both directions; with n = 12 per workflow that is noise, not a ranking. Both trail the best LLMs, which
  reason over the whole case — the price of a decision model, real or local.
- **Speed: 10–90× slower than Jev.** A hybrid (Gated DeltaNet) 27B on 24 GB caches its recurrent state in
  1,568-token pages, so vLLM runs a 9k-token case one sequence at a time. invoice_processing — 55 questions
  over a ~9k-token state — is the worst case. Jev runs on TypeSafe's cluster.
- **Cost**: usage counts the state once per request, as TypeSafe bills; the estimates the harness prints use
  Jev's price list and are meaningless for a local GPU.

Agreement against time per case, published runs and e-jev's:

![invoice_processing](evals/invoice_processing.svg)
![customer_service](evals/customer_service.svg)
![agent_trace_observability](evals/agent_trace_observability.svg)
![security_incidents](evals/security_incidents.svg)
