"""HTTP middleware shared by the full service and the serverless function.

FastAPI's ``APIRoute`` does not register ``HEAD`` alongside ``GET`` (unlike Starlette's
plain ``Route``), so ``HEAD /api/v1/health`` answered 405. Uptime monitors and link
checkers probe with HEAD, and a 405 reads as "service broken".

Handling it once in middleware is better than decorating every route: the behaviour
becomes uniform, and it stays correct for routes added later.
"""
from __future__ import annotations

from typing import Any, Callable, Awaitable


class HeadAsGetMiddleware:
    """Serve ``HEAD`` by running the ``GET`` handler and discarding the body.

    Implemented at the ASGI level rather than with ``BaseHTTPMiddleware`` because the
    latter wraps every response in a streaming object, where blanking ``.body`` has no
    effect. Response headers — including ``Content-Length`` — pass through unchanged,
    which is what RFC 9110 requires for HEAD.
    """

    def __init__(self, app: Callable[..., Awaitable[Any]]) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or scope.get("method") != "HEAD":
            await self.app(scope, receive, send)
            return

        # Copy so the original scope is not mutated for anything downstream.
        scope = dict(scope)
        scope["method"] = "GET"

        async def send_without_body(message: dict) -> None:
            if message.get("type") == "http.response.body":
                message = {**message, "body": b"", "more_body": False}
            await send(message)

        await self.app(scope, receive, send_without_body)
