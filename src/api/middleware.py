import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

logger = logging.getLogger("api")


async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Registra requisições e calcula a latência."""
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        _log_error(request, (time.perf_counter() - start_time) * 1000)
        raise

    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "%s %s | status=%s | latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    return response


def _log_error(request: Request, latency_ms: float) -> None:
    """Loga uma requisição HTTP que terminou em exceção."""
    logger.exception(
        "ERRO | %s %s | latency_ms=%.2f",
        request.method,
        request.url.path,
        latency_ms,
    )
