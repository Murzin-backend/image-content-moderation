"""Контроллер проверки здоровья — GET /api/health."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.serializers.responses import HealthResponse
from app.core.logging import get_logger
from app.core.services.analysis import AnalysisService

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
)
async def health_check(request: Request) -> HealthResponse:
    """Вернуть статус сервиса и состояние загрузки ML-моделей."""
    analysis_service: AnalysisService | None = getattr(
        request.app.state, "analysis_service", None,
    )
    models_loaded = analysis_service.models_loaded if analysis_service else False
    status = "healthy" if models_loaded else "degraded"

    return HealthResponse(status=status, models_loaded=models_loaded)
