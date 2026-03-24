"""Агрегатор результатов — объединяет обнаружения символов и текста в вердикт."""

from __future__ import annotations

from app.core.logging import get_logger
from app.core.models import AnalysisResult, SymbolDetection, TextDetection, Verdict

logger = get_logger(__name__)

_TEXT_CONFIDENCE_WEIGHT: float = 0.85


class ResultAggregator:
    """Агрегатор без состояния: объединяет обнаружения и вычисляет итоговый вердикт."""

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        suspicious_threshold: float = 0.35,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._suspicious_threshold = suspicious_threshold

    def aggregate(
        self,
        symbol_detections: list[SymbolDetection],
        text_detections: list[TextDetection],
    ) -> AnalysisResult:
        """Вычислить итоговый score и verdict на основе обнаружений.

        Стратегия:
        1. Максимальная уверенность из обнаружений символов.
        2. Максимальная уверенность из текстовых обнаружений, взвешенная.
        3. score = max(symbol_max, text_max * weight).
        4. Преобразование score → verdict по настроенным порогам.
        """
        symbol_max = max(
            (d.confidence for d in symbol_detections),
            default=0.0,
        )
        text_max = max(
            (d.confidence for d in text_detections),
            default=0.0,
        )
        weighted_text = text_max * _TEXT_CONFIDENCE_WEIGHT
        score = round(max(symbol_max, weighted_text), 4)

        verdict = self._score_to_verdict(score)

        logger.info(
            "aggregation_complete",
            symbol_max=symbol_max,
            text_max=text_max,
            score=score,
            verdict=verdict.value,
        )

        return AnalysisResult(
            verdict=verdict,
            score=score,
            symbol_detections=list(symbol_detections),
            text_detections=list(text_detections),
        )

    def _score_to_verdict(self, score: float) -> Verdict:
        """Преобразовать числовой score в вердикт."""
        if score >= self._confidence_threshold:
            return Verdict.REJECTED
        if score >= self._suspicious_threshold:
            return Verdict.SUSPICIOUS
        return Verdict.CLEAN
