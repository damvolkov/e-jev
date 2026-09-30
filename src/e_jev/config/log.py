"""config.log: renderer and level."""

from typing import Literal

from pydantic_settings import BaseSettings


class LogSettings(BaseSettings):
    env: Literal["prod", "dev"] = "prod"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
