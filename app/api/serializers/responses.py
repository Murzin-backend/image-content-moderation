"""Схемы ответов API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.models import Verdict


class SymbolDetectionItem(BaseModel):
    """Результат обнаружения символики."""

    label: str = Field(..., description="Метка обнаруженного символа", examples=["rainbow flag"])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели", examples=[0.87])


class TextDetectionItem(BaseModel):
    """Результат обнаружения текстового маркера через OCR."""

    keyword: str = Field(..., description="Найденное ключевое слово", examples=["pride"])
    context: str = Field(..., description="Контекст вокруг найденного слова", examples=["...happy pride month..."])
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели", examples=[0.95])


class AnalysisResponse(BaseModel):
    """Ответ на запрос анализа изображения."""

    verdict: Verdict = Field(..., description="Итоговый вердикт модерации", examples=[Verdict.REJECTED])
    score: float = Field(..., ge=0.0, le=1.0, description="Итоговый показатель уверенности", examples=[0.87])
    symbol_detections: list[SymbolDetectionItem] = Field(
        default_factory=list, description="Список обнаруженных символов",
    )
    text_detections: list[TextDetectionItem] = Field(
        default_factory=list, description="Список обнаруженных текстовых маркеров",
    )
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах", examples=[342.0])


class HealthResponse(BaseModel):
    """Ответ на запрос проверки здоровья сервиса."""

    status: str = Field(..., description="Статус сервиса", examples=["healthy"])
    models_loaded: bool = Field(..., description="Загружены ли ML-модели", examples=[True])


class ErrorResponse(BaseModel):
    """Схема ответа об ошибке."""

    detail: str = Field(..., description="Описание ошибки", examples=["Файл слишком большой"])
