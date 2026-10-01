##### CONFIG #####
PROJECT  ?= e-jev
IMAGE    ?= ghcr.io/damvolkov/e-jev:latest
COMPOSE  ?= $(HOME)/.config/compose
UNITS    ?= $(HOME)/.config/systemd/user
SERVICES ?= vllm jev
TYPESAFE_NODE ?= @typesafe-ai/n8n-nodes-typesafe-ai@0.9.0
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
.PHONY: help sync lint type arch test check build deploy ui watch restart status logs calibrate

help:
	@printf "$(BOLD)$(CYAN)$(PROJECT)$(RESET) $(GRAY)· typed decisions over vLLM · compose + systemd --user$(RESET)\n\n"
	@awk 'BEGIN{FS=":.*##"} /^[a-z][a-zA-Z0-9_-]*:.*##/{printf "  $(GREEN)%-10s$(RESET) $(GRAY)%s$(RESET)\n",$$1,$$2}' $(MAKEFILE_LIST)

sync: ## sync deps (main + dev)
	@uv sync

lint: ## ruff check + format check
	@uv run ruff check src tests && uv run ruff format --check src tests

type: ## ty static types
	@uv run ty check

arch: ## tach layer boundaries
	@uv run tach check

test: ## pytest
	@uv run pytest $(ARGS)

check: lint type arch test ## every local gate

build: ## build the jev image
	@docker build -t $(IMAGE) .

deploy: build ## install compose dirs + systemd units, enable and (re)start vllm + jev
	@for s in $(SERVICES); do mkdir -p $(COMPOSE)/$$s && cp deploy/$$s/compose.yml $(COMPOSE)/$$s/compose.yml && cp deploy/systemd/$$s.service $(UNITS)/$$s.service; done
	@docker run --rm -v $(COMPOSE)/data:/d busybox sh -c 'mkdir -p /d/jev && chmod 777 /d/jev'
	@systemctl --user daemon-reload
	@systemctl --user enable $(addsuffix .service,$(SERVICES))
	@systemctl --user restart $(addsuffix .service,$(SERVICES))
	@printf "$(GREEN)✓ deployed: $(SERVICES)$(RESET)\n"

ui: ## n8n + official TypeSafe node, credential on jev, demo workflow (http://localhost:45700)
	@mkdir -p $(COMPOSE)/n8n && cp deploy/n8n/compose.yml $(COMPOSE)/n8n/compose.yml && cp deploy/systemd/n8n.service $(UNITS)/n8n.service
	@docker run --rm -v $(COMPOSE)/data:/d busybox sh -c 'mkdir -p /d/n8n && chown 1000:1000 /d/n8n'
	@systemctl --user daemon-reload && systemctl --user enable n8n.service && systemctl --user restart n8n.service
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' n8n 2>/dev/null)" = healthy ]; do sleep 3; done
	@docker exec n8n sh -c 'mkdir -p ~/.n8n/nodes && cd ~/.n8n/nodes && npm install --no-fund --no-audit --silent --legacy-peer-deps $(TYPESAFE_NODE)'
	@docker cp deploy/n8n/credentials.json n8n:/tmp/credentials.json && docker cp deploy/n8n/workflow.json n8n:/tmp/workflow.json
	@docker exec n8n n8n import:credentials --input=/tmp/credentials.json && docker exec n8n n8n import:workflow --input=/tmp/workflow.json
	@systemctl --user restart n8n.service
	@printf "$(GREEN)✓ n8n on http://localhost:45700 — workflow 'e-jev · System One playground'$(RESET)\n"

watch: ## Phoenix: traces UI + OTLP sink for jev and vllm (http://localhost:45900)
	@mkdir -p $(COMPOSE)/phoenix && cp deploy/phoenix/compose.yml $(COMPOSE)/phoenix/compose.yml && cp deploy/systemd/phoenix.service $(UNITS)/phoenix.service
	@docker run --rm -v $(COMPOSE)/data:/d busybox sh -c 'mkdir -p /d/phoenix'
	@systemctl --user daemon-reload && systemctl --user enable phoenix.service && systemctl --user restart phoenix.service
	@until [ "$$(docker inspect -f '{{.State.Health.Status}}' phoenix 2>/dev/null)" = healthy ]; do sleep 3; done
	@printf "$(GREEN)✓ phoenix on http://localhost:45900$(RESET)\n"

restart: ## restart vllm + jev
	@systemctl --user restart $(addsuffix .service,$(SERVICES))

status: ## systemd + container health
	@systemctl --user --no-pager status $(addsuffix .service,$(SERVICES)) | grep -E '●|Active'
	@docker ps --filter name='^(vllm|jev)$$' --format '{{.Names}}\t{{.Status}}'

logs: ## follow logs: make logs [vllm|jev]
	@journalctl --user -fu $(or $(ARGS),jev).service

calibrate: ## fit temperature: make calibrate labeled.jsonl (copied into data/jev), then restart jev
	@cp $(ARGS) $(COMPOSE)/data/jev/labeled.jsonl
	@docker exec jev python -m e_jev.cli.calibrate /data/labeled.jsonl
	@systemctl --user restart jev.service

%:
	@:
