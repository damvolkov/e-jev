"""Integration fixtures: a live e-jev (or Jev) endpoint, or the whole tier is skipped."""

import httpx
import pytest

from examples.triage.graph import TriageSettings


@pytest.fixture(scope="session")
def triage_settings() -> TriageSettings:
    settings = TriageSettings()
    try:
        ready = httpx.get(f"{settings.url}/ready", timeout=5).is_success
    except httpx.HTTPError:
        ready = False
    match ready:
        case True:
            return settings
        case False:
            pytest.skip(f"no ready System One endpoint at {settings.url}")
