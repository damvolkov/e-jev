"""config.cli: where `ejev` sends its questions unless a flag says otherwise."""

from pydantic_settings import BaseSettings


class CliSettings(BaseSettings):
    ### Any System One endpoint: this e-jev, or TypeSafe's https://api.typesafe.ai with a real key.
    cli_url: str = "http://localhost:45160"
    cli_key: str = "local"
    cli_model: str = "jev-latest"
