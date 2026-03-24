"""Фабрика приложения FastAPI с управлением жизненным циклом."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.exceptions import register_exception_handlers
from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.pipeline.aggregator import ResultAggregator
from app.core.services.analysis import AnalysisService
from app.core.services.symbol_detector import SymbolDetector
from app.core.services.text_detector import TextDetector

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Управление жизненным циклом: загрузка и выгрузка ML-моделей."""
    settings = get_settings()

    setup_logging(log_level=settings.LOG_LEVEL, json_output=not settings.DEBUG)

    logger.info(
        "starting_application",
        app_name=settings.APP_NAME,
        clip_model=settings.CLIP_MODEL_NAME,
        device=settings.clip_device_resolved,
    )

    symbol_detector = SymbolDetector(
        model_name=settings.CLIP_MODEL_NAME,
        device=settings.clip_device_resolved,
    )
    symbol_detector.load()

    text_detector = TextDetector(languages=settings.easyocr_language_list)
    text_detector.load()

    aggregator = ResultAggregator(
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        suspicious_threshold=settings.SUSPICIOUS_THRESHOLD,
    )

    analysis_service = AnalysisService(
        symbol_detector=symbol_detector,
        text_detector=text_detector,
        aggregator=aggregator,
        max_concurrent=settings.MAX_CONCURRENT_ANALYSES,
    )

    app.state.analysis_service = analysis_service

    logger.info("application_ready")

    yield

    logger.info("shutting_down")


def create_app() -> FastAPI:
    """Фабрика приложения — возвращает настроенный экземпляр FastAPI."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="Сервис модерации изображений",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    app.include_router(api_router)
    register_exception_handlers(app)

    return app
