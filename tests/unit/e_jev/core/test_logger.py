import msgspec
import pytest
import structlog

from e_jev.core.logger import setup_logger


async def test_logger_prod_emits_json(capfdbinary: pytest.CaptureFixture[bytes]) -> None:
    setup_logger("prod", "INFO")
    structlog.get_logger().info("reader_opened", model="m")
    event = msgspec.json.decode(capfdbinary.readouterr().out.splitlines()[-1])
    assert (event["event"], event["model"], event["level"]) == ("reader_opened", "m", "info")


async def test_logger_filters_below_level(capfd: pytest.CaptureFixture[str]) -> None:
    setup_logger("dev", "WARNING")
    structlog.get_logger().info("quiet")
    assert "quiet" not in capfd.readouterr().out
