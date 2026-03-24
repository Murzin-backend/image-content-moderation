"""Обработчики исключений и утилиты для OpenAPI-документации."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.serializers.responses import ErrorResponse
from app.core.logging import get_logger
from app.exceptions import AppException

logger = get_logger(__name__)


def collect_responses(*exc_classes: type[AppException]) -> dict:
    """Собрать схемы ответов из классов исключений для OpenAPI-документации."""
    return {
        exc.status_code: {"model": ErrorResponse, "description": exc.description}
        for exc in exc_classes
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Зарегистрировать обработчики исключений в приложении FastAPI."""

    @app.exception_handler(AppException)
    async def _app_exception_handler(
        request: Request,
        exc: AppException,
    ) -> JSONResponse:
        logger.warning("app_exception", status_code=exc.status_code, detail=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=exc.detail).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception("unhandled_exception", error=str(exc))
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(detail="Внутренняя ошибка сервера").model_dump(),
        )
