"""api.handlers: the failure tree mapped to TypeSafe's status codes, at the boundary only."""

from typing import Final

import structlog
from litestar import Request, Response
from litestar.exceptions import ValidationException
from litestar.status_codes import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_502_BAD_GATEWAY
from litestar.types import ExceptionHandlersMap

from e_jev.core.errors import STATUSES, RequestError, RetryableError, ValidationError

log = structlog.get_logger()


def handle_invalid(_: Request, error: ValidationException) -> Response:
    """Request validation as TypeSafe reports it: 422 with a `detail` list."""
    return Response(
        {"detail": [{"loc": ["body"], "msg": error.detail, "type": "value_error", "ctx": error.extra}]},
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
    )


def handle_unprocessable(_: Request, error: ValidationError) -> Response:
    return Response({"detail": [{"loc": ["body"], "msg": error.message, "type": "value_error"}]}, status_code=HTTP_422_UNPROCESSABLE_ENTITY)


def handle_capacity(_: Request, error: RetryableError) -> Response:
    """Capacity failures leave as the statuses the official SDKs retry: 429 with its hint, or Jev's 529."""
    headers = {"retry-after": str(int(error.retry_after))} if error.retry_after is not None else {}
    return Response({"detail": error.message}, status_code=STATUSES[error.capacity], headers=headers)


def handle_upstream(request: Request, error: RequestError) -> Response:
    log.error("upstream_failed", path=request.url.path, error=error.message, exc_info=error)
    return Response({"detail": error.message}, status_code=HTTP_502_BAD_GATEWAY)


HANDLERS: Final[ExceptionHandlersMap] = {
    ValidationException: handle_invalid,
    ValidationError: handle_unprocessable,
    RetryableError: handle_capacity,
    RequestError: handle_upstream,
}
