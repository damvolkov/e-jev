##### CONFIG #####
PROJECT  ?= e-jev
IMAGE    ?= ghcr.io/damvolkov/e-jev:latest
COMPOSE  ?= $(HOME)/.config/compose
UNITS    ?= $(HOME)/.config/systemd/user
SERVICES ?= ejev vllm clef
TYPESAFE_NODE ?= @typesafe-ai/n8n-nodes-typesafe-ai@0.9.0
EVALS_REV ?= 0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb
EVALS_LIMIT ?= 12
JEV_URL ?= http://localhost:45160
CALIBRATION_SET ?= .cache/labeled.jsonl
CALIBRATION_SIZE ?= 300
JEV_TEST_BODY ?= {"model":"jev-latest","state":"Me han cobrado dos veces la factura de marzo. Quiero el reembolso o cancelo.","questions":{"dep":{"type":"choice","instructions":"¿Qué equipo lo gestiona?","criteria":{"billing":"facturas, pagos","tech":"errores","sales":"precios"}},"urgente":{"type":"noul","instructions":"¿es urgente?"},"enfadado":{"type":"score","instructions":"¿Qué tan enfadado está el cliente?","criteria":["calmado","molesto","furious"]}}}
EVALS_DIR := .cache/workflowevals
WORKFLOWS := invoice_processing customer_service agent_trace_observability security_incidents
ARGS      = $(filter-out $(firstword $(MAKECMDGOALS)),$(MAKECMDGOALS))

RESET   := \033[0m
BOLD    := \033[1m
GREEN   := \033[0;32m
CYAN    := \033[0;36m
GRAY    := \033[0;90m

SHELL := bash
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

##### TARGETS #####
.PHONY: help sync lint type arch test integration graph evals ejev-test check build deploy ui watch restart status logs calibrate

help:
	@printf "$(BOLD)$(CYAN)$(PROJECT)$(RESET) $(GRAY)· typed decisions over vLLM · compose + systemd --user$(RESET)\n\n"
	@awk 'BEGIN{FS=":.*##"} /^[a-z][a-zA-Z0-9_-]*:.*##/{printf "  $(GREEN)%-10s$(RESET) $(GRAY)%s$(RESET)\n",$$1,$$2}' $(MAKEFILE_LIST)

sync: ## sync deps (main + dev)
	@uv sync

lint: ## ruff check + format check
	@uv run ruff check src tests examples && uv run ruff format --check src tests examples

type: ## ty static types
	@uv run ty check

arch: ## tach layer boundaries
	@uv run tach check

test: ## pytest
	@uv run pytest $(ARGS)

integration: ## live tests against JEV_URL (default http://localhost:45160): the triage graph through Pydantic AI
	@uv run pytest -m integration $(ARGS)

ejev-test: ## the service in one shot: ready, one noul + one choice + one score, timed (JEV_URL)
	@curl -fsS $(JEV_URL)/ready >/dev/null && printf "$(GREEN)✓ ready$(RESET) "
	@body=$$(mktemp); t0=$$(date +%s%N); curl -fsS -o $$body -X POST $(JEV_URL)/v1/systemone -H 'content-type: application/json' -d '$(JEV_TEST_BODY)'; ms=$$(( ( $$(date +%s%N) - t0 ) / 1000000 )); command -v jq >/dev/null && jq . $$body || cat $$body; rm -f $$body; printf "$(GRAY)· %s ms · $(JEV_URL)$(RESET)\n" "$$ms"

graph: ## run the triage graph on one ticket: make graph "text"
	@PYDANTIC_AI_NO_BANNER=1 uv run python -m examples.triage.graph "$(ARGS)"

evals: ## TypeSafe's WorkflowEvals against e-jev (EVALS_LIMIT cases per workflow), scored against published Jev + LLMs
	@[ -d $(EVALS_DIR) ] || git clone -q https://github.com/typesafe-ai/WorkflowEvals $(EVALS_DIR)
	@git -C $(EVALS_DIR) checkout -q $(EVALS_REV) && cd $(EVALS_DIR) && uv sync --locked -q
	@cd $(EVALS_DIR) && for w in $(WORKFLOWS); do TYPESAFE_API_KEY=local uv run python run.py $$w --base-url $(JEV_URL) --name e-jev --limit $(EVALS_LIMIT) --workers 2 | tail -2; done
	@cd $(EVALS_DIR) && uv run --extra plot python $(CURDIR)/scripts/evals_compare.py

check: lint type arch test ## every local gate

build: ## build the ejev image
	@docker build -t $(IMAGE) .

deploy: build ## install compose dirs + systemd units, enable and (re)start the stack
	@for s in $(SERVICES); do mkdir -p $(COMPOSE)/$$s && cp deploy/$$s/compose.yml $(COMPOSE)/$$s/compose.yml && cp deploy/systemd/$$s.service $(UNITS)/$$s.service; done
	@docker run --rm -v $(COMPOSE)/data:/d busybox sh -c 'mkdir -p /d/ejev && chmod 777 /d/ejev'
	@systemctl --user daemon-reload
	@systemctl --user enable $(addsuffix .service,$(SERVICES))
	@systemctl --user restart $(addsuffix .service,$(SERVICES))
	@printf "$(GREEN)✓ deployed: $(SERVICES)$(RESET)\n"

ui: ## n8n + official TypeSafe node, credential on ejev, demo workflow (http://localhost:45700)
	@mkdir -p $(COMPOSE)/n8n && cp deploy/n8n/compose.yml $(COMPOSE)/n8n/compose.yml && cp deploy/systemd/n8n.service $(UNITS)/n8n.service
	@docker run --rm -v $(COMPOSE)/data:/d busybox sh -c 'mkdir -p /d/n8n && chown 1000:1000 /d/n8n'
	@systemctl --user daemon-reload && systemctl --user enable n8n.service && systemctl --user restart n8n.service
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' n8n 2>/dev/null)" = healthy ]; do sleep 3; done
	@docker exec n8n sh -c 'mkdir -p ~/.n8n/nodes && cd ~/.n8n/nodes && npm install --no-fund --no-audit --silent --legacy-peer-deps $(TYPESAFE_NODE)'
	@docker cp deploy/n8n/credentials.json n8n:/tmp/credentials.json && docker cp deploy/n8n/workflow.json n8n:/tmp/workflow.json
	@docker exec n8n n8n import:credentials --input=/tmp/credentials.json && docker exec n8n n8n import:workflow --input=/tmp/workflow.json
	@systemctl --user restart n8n.service
	@printf "$(GREEN)✓ n8n on http://localhost:45700 — workflow 'e-jev · System One playground'$(RESET)\n"

watch: ## Phoenix: traces UI + OTLP sink for ejev and vllm (http://localhost:45900)
	@mkdir -p $(COMPOSE)/phoenix && cp deploy/phoenix/compose.yml $(COMPOSE)/phoenix/compose.yml && cp deploy/systemd/phoenix.service $(UNITS)/phoenix.service
	@docker run --rm -v $(COMPOSE)/data:/d busybox sh -c 'mkdir -p /d/phoenix'
	@systemctl --user daemon-reload && systemctl --user enable phoenix.service && systemctl --user restart phoenix.service
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' phoenix 2>/dev/null)" = healthy ]; do sleep 3; done
	@printf "$(GREEN)✓ phoenix on http://localhost:45900$(RESET)\n"

strata: ## install the Strata compose dir + unit, never enabled: the card owner is decided by gpu-*
	@mkdir -p $(COMPOSE)/strata && cp deploy/strata/compose.yml $(COMPOSE)/strata/compose.yml && cp deploy/systemd/strata.service $(UNITS)/strata.service
	@docker run --rm -v $(COMPOSE)/data:/d busybox sh -c 'mkdir -p /d/strata'
	@systemctl --user daemon-reload
	@printf "$(GREEN)✓ strata unit installed (disabled) — make gpu-human to hand it the card$(RESET)\n"

gpu-decide: ## default serving pattern: decision model on the card, teacher stopped (clef + ejev)
	@systemctl --user stop vllm.service strata.service
	@systemctl --user start clef.service ejev.service
	@printf "$(GREEN)✓ clef + ejev up — make gpu-teach for the vLLM teacher, gpu-human for Strata chat$(RESET)\n"

gpu-teach: ## teacher pattern: vLLM on the card for extract/long context/fitting, decision model off
	@systemctl --user stop clef.service strata.service
	@systemctl --user start vllm.service ejev.service
	@printf "$(GREEN)✓ vllm + ejev up$(RESET)\n"

gpu-human: ## Strata on the card: 125B chat for webui/coders/cortexdb; jev stays up, its GPU routes degrade by design
	@systemctl --user stop vllm.service clef.service
	@systemctl --user start strata.service ejev.service
	@printf "$(GREEN)✓ strata + jev up — webui on http://localhost:45300, OpenAI base http://strata:8080/v1$(RESET)\n"

restart: ## restart the fleet
	@systemctl --user restart $(addsuffix .service,$(SERVICES))

status: ## systemd + container health
	@systemctl --user --no-pager status $(addsuffix .service,$(SERVICES)) | grep -E '●|Active'
	@docker ps --filter name='^(vllm|ejev|clef)$$' --format '{{.Names}}\t{{.Status}}'

logs: ## follow logs: make logs [vllm|jev]
	@journalctl --user -fu $(or $(ARGS),ejev).service

calibrate: ## fit per-kind temperatures: make calibrate [labeled.jsonl] (default: BoolQ, AG News, Banking77, SST-5), then restart ejev
	@[ -n "$(ARGS)" ] || uv run --with datasets python scripts/calibration_build.py $(CALIBRATION_SET) $(CALIBRATION_SIZE)
	@JEV_CALIBRATION=$(COMPOSE)/data/ejev/calibration.json uv run ejev calibrate $(or $(ARGS),$(CALIBRATION_SET))
	@systemctl --user restart ejev.service

%:
	@:
