"""Контроллер анализа изображений — POST /api/analyze."""

from __future__ import annotations

from fastapi import APIRouter, Request, UploadFile, File

from app.api.exceptions import collect_responses
from app.api.serializers.responses import AnalysisResponse, SymbolDetectionItem, TextDetectionItem
from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.services.analysis import AnalysisService
from app.core.utils.image import load_image, resolve_content_type, validate_content_type, validate_file_size
from app.exceptions import (
    ImageDecodeError,
    ModelNotReadyError,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
)

logger = get_logger(__name__)

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses=collect_responses(
        PayloadTooLargeError,
        UnsupportedMediaTypeError,
        ImageDecodeError,
        ModelNotReadyError,
    ),
    summary="Анализ изображения на наличие запрещённой символики и текстовых маркеров",
)
async def analyze_image(
    request: Request,
    file: UploadFile = File(..., description="Файл изображения (JPEG, PNG, WebP, BMP, TIFF)"),
) -> AnalysisResponse:
    """Принять изображение через multipart/form-data и вернуть результаты модерации."""
    settings = get_settings()

    data = await file.read()
    validate_file_size(data, settings.max_image_bytes)

    resolved_type = resolve_content_type(file.content_type, data, file.filename)
    validate_content_type(resolved_type, settings.ALLOWED_CONTENT_TYPES)

    image = load_image(data)

    analysis_service: AnalysisService = request.app.state.analysis_service

    if not analysis_service.models_loaded:
        raise ModelNotReadyError()

    result = await analysis_service.analyze(image)

    return AnalysisResponse(
        verdict=result.verdict,
        score=result.score,
        symbol_detections=[
            SymbolDetectionItem(label=d.label, confidence=d.confidence)
            for d in result.symbol_detections
        ],
        text_detections=[
            TextDetectionItem(
                keyword=d.keyword,
                context=d.context,
                confidence=d.confidence,
            )
            for d in result.text_detections
        ],
        processing_time_ms=result.processing_time_ms,
    )
