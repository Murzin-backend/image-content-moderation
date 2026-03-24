"""DTO для результатов инференса ML-моделей."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.models.domain import SymbolDetection, TextDetection


@dataclass(frozen=True)
class ClipInferenceResult:
    """Результат инференса CLIP-детектора."""

    detections: list[SymbolDetection] = field(default_factory=list)
    best_score: float = 0.0
    best_label: str | None = None
    positive_similarities: list[float] = field(default_factory=list)
    negative_max: float = 0.0


@dataclass(frozen=True)
class OcrInferenceResult:
    """Результат инференса OCR-детектора."""

    detections: list[TextDetection] = field(default_factory=list)
    full_text_length: int = 0

