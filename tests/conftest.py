"""Общие фикстуры для тестов."""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.models import AnalysisResult, Verdict
from app.core.services.analysis import AnalysisService


@pytest.fixture()
def mock_analysis_service() -> AnalysisService:
    """Вернуть мок AnalysisService, который всегда возвращает CLEAN."""
    service = MagicMock(spec=AnalysisService)
    service.models_loaded = True

    async def _analyze(image):  # noqa: ANN001
        return AnalysisResult(
            verdict=Verdict.CLEAN,
            score=0.05,
            symbol_detections=[],
            text_detections=[],
            processing_time_ms=10.0,
        )

    service.analyze = _analyze
    return service


@pytest.fixture()
def app(mock_analysis_service: AnalysisService) -> FastAPI:
    """Создать тестовое приложение FastAPI с замоканным ML-сервисом."""
    from app.main import create_app

    test_app = create_app()
    test_app.state.analysis_service = mock_analysis_service
    return test_app


@pytest.fixture()
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Асинхронный HTTP-клиент для тестов."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
