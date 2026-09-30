"""api.security: TypeSafe's bearer contract — enforced only when an API key is configured."""

from typing import Final

from litestar.connection import ASGIConnection
from litestar.exceptions import NotAuthorizedException
from litestar.handlers import BaseRouteHandler

REFUSED: Final = "Missing or invalid API key."


def bearer(connection: ASGIConnection, _: BaseRouteHandler) -> None:
    match connection.app.state.settings.api_key:
        case None:
            return
        case expected if connection.headers.get("authorization") == f"Bearer {expected}":
            return
        case _:
            raise NotAuthorizedException(REFUSED)
