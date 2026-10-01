"""api.middlewares: the ASGI chain — every response carries the request id TypeSafe's SDKs read."""

from functools import partial
from typing import Final
from uuid import uuid4

import structlog
from litestar.datastructures import MutableScopeHeaders
from litestar.enums import ScopeType
from litestar.middleware import ASGIMiddleware
from litestar.types import ASGIApp, Message, Receive, Scope, Send

HEADER: Final = "x-typesafe-request-id"


class RequestIdMiddleware(ASGIMiddleware):
    """One id per request: on the response header, and on every log line the request emits."""

    scopes = (ScopeType.HTTP,)

    ##### PRIVATE #####

    @staticmethod
    async def _handle_send(send: Send, request_id: str, message: Message) -> None:
        match message["type"]:
            case "http.response.start":
                MutableScopeHeaders.from_message(message)[HEADER] = request_id
            case _:
                pass
        await send(message)

    ############################################################

    ##### PUBLIC #####

    async def handle(self, scope: Scope, receive: Receive, send: Send, next_app: ASGIApp) -> None:
        request_id = uuid4().hex
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            await next_app(scope, receive, partial(self._handle_send, send, request_id))


MIDDLEWARES: Final = (RequestIdMiddleware(),)
