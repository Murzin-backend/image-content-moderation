"""Пакет моделей — реэкспорт доменных моделей и DTO."""

from app.core.models.domain import (
    AnalysisResult,
    SymbolDetection,
    TextDetection,
    Verdict,
)
from app.core.models.dto import ClipInferenceResult, OcrInferenceResult

__all__ = [
    "AnalysisResult",
    "ClipInferenceResult",
    "OcrInferenceResult",
    "SymbolDetection",
    "TextDetection",
    "Verdict",
]
