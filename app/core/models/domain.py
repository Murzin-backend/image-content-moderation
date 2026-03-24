"""Доменные модели (объекты-значения), используемые в пайплайне."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Verdict(str, enum.Enum):
    """Итоговый вердикт модерации."""

    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class SymbolDetection:
    """Результат обнаружения символа/эмблемы через CLIP."""

    label: str
    confidence: float


@dataclass(frozen=True)
class TextDetection:
    """Результат совпадения ключевого слова через OCR."""

    keyword: str
    context: str
    confidence: float


@dataclass
class AnalysisResult:
    """Агрегированный результат анализа."""

    verdict: Verdict
    score: float
    symbol_detections: list[SymbolDetection] = field(default_factory=list)
    text_detections: list[TextDetection] = field(default_factory=list)
    processing_time_ms: float = 0.0
