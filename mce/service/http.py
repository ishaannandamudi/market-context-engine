"""Small stdlib WSGI adapter for the framework-neutral service."""

from collections.abc import Callable, Iterable
from typing import Any

from mce.orchestration.models import PipelineErrorCode
from mce.service.api import MAX_REQUEST_BYTES, MarketContextService

_REASONS = {
    200: "OK",
    404: "Not Found",
    413: "Payload Too Large",
    422: "Unprocessable Entity",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


class MarketContextWSGI:
    def __init__(self, service: MarketContextService) -> None:
        self.service = service

    def __call__(
        self,
        environ: dict[str, Any],
        start_response: Callable[[str, list[tuple[str, str]]], Any],
    ) -> Iterable[bytes]:
        try:
            length = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_REQUEST_BYTES:
            status, body = self.service._error(
                413,
                PipelineErrorCode.INVALID_REQUEST,
                "request payload too large",
            )
        else:
            stream = environ.get("wsgi.input")
            payload = stream.read(length) if stream is not None else b""
            status, body = self.service.handle(
                str(environ.get("REQUEST_METHOD", "GET")),
                str(environ.get("PATH_INFO", "/")),
                payload,
            )
        start_response(
            f"{status} {_REASONS[status]}",
            [
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]
