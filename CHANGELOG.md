# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Fleet stability: a backend that failed to open (the vLLM reader caught mid-model-load is the case)
  reopens itself once the server returns — retried at most once per `REOPEN_DELAY` with its freshest
  diagnosis — instead of parking `/ready` at 503 for the container's whole life; the reader's open
  classifies HTTP refusals instead of collapsing them into `unreachable`.
- Deploy: systemd is the sole owner of the ejev container (`restart: "no"`) and the unit gates on vLLM
  health — it waits while the vLLM container is `starting` and passes when it is absent, so the
  daemon's restart manager can no longer revive ejev beside a still-loading model (the 503 poisoning
  after a dockerd crash) and `gpu-decide` still serves without vLLM.

### Changed

- Identity: the multi-backend purpose is stated in the description and README — the decision backend is a
  routable fleet (Clef, the vLLM reader, Laya, any `/v1/systemone` server), not a single model.
- `pyproject.toml`: `[tool.edamlabs]` declares `kind = "service"` and `family = "eager"`; authors move to
  EDamLabs; description no longer anchored on vLLM.
- README: header standardized on the EDamLabs template (badge row, banner wordmark without an `<h1>`,
  identity line).
- `CHANGELOG.md` added: the fleet's release and drift modules read it.

## [0.2.0] - 2026-10-14

### Added

- Backend fleet: `JEV_BACKENDS` / `JEV_ROWS` route per question kind, language and state size, falling
  down the row on failure (`BackendKind`: reading, systemone, laya).
- Clef-Flash as a trained decision backend; the reader as the logprob teacher and sole extractor.
- `ejev` CLI (ask, calibrate) and `ejev-mcp` (the same decisions as MCP tools).
- TypeSafe wire compatibility: `POST /v1/systemone`, `POST /v1/extract`, `GET /v1/models`; capacity
  answers 429/529.
- Per-(backend, kind) calibration with a stable fit/held-out split over BoolQ, AG News, Banking77, SST-5
  and a Spanish set.
- systemd --user and Docker deployment (`make deploy`, `gpu-decide`, `gpu-teach`).
