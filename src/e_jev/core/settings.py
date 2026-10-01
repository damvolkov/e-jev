"""core.settings: every domain's settings composed into one contract, read once from `JEV_*`."""

from pydantic_settings import SettingsConfigDict

from e_jev.config.log import LogSettings
from e_jev.config.systemone import SystemOneSettings
from e_jev.config.telemetry import TelemetrySettings
from e_jev.config.vllm import VllmSettings


class Settings(VllmSettings, SystemOneSettings, TelemetrySettings, LogSettings):
    model_config = SettingsConfigDict(env_prefix="JEV_", frozen=True)


settings = Settings()
