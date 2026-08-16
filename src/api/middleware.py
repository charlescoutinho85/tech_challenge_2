import logging
import time

from fastapi import Request


logger = logging.getLogger("api")


async def log_requests(request: Request, call_next):
    """Registra requisições e calcula a latência."""

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        logger.info(
            "%s %s | status=%s | latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )

        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"

        return response

    except Exception:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        logger.exception(
            "ERRO | %s %s | latency_ms=%.2f",
            request.method,
            request.url.path,
            latency_ms,
        )

        raise