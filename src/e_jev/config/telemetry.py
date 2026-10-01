"""config.telemetry: where traces go — unset keeps tracing off."""

from pydantic_settings import BaseSettings


class TelemetrySettings(BaseSettings):
    ### OTLP/HTTP traces endpoint, e.g. http://phoenix:6006/v1/traces; also the Phoenix project name below.
    otlp_endpoint: str | None = None
    otlp_project: str = "e-jev"
