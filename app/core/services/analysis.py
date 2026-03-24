"""AnalysisService — оркестратор полного пайплайна анализа изображений."""

from __future__ import annotations

import asyncio
import time

from PIL import Image

from app.core.logging import get_logger
from app.core.models import AnalysisResult
from app.core.models.dto import ClipInferenceResult, OcrInferenceResult
from app.core.pipeline.aggregator import ResultAggregator
from app.core.services.symbol_detector import SymbolDetector
from app.core.services.text_detector import TextDetector

logger = get_logger(__name__)


class AnalysisService:
    """Оркестратор: запускает детекторы параллельно и агрегирует результаты.

    CLIP и EasyOCR — CPU/GPU-bound операции, поэтому инференс
    выполняется в пуле потоков через asyncio.to_thread.
    Семафор ограничивает количество одновременных инференсов.
    """

    def __init__(
        self,
        symbol_detector: SymbolDetector,
        text_detector: TextDetector,
        aggregator: ResultAggregator,
        max_concurrent: int = 4,
    ) -> None:
        self._symbol_detector = symbol_detector
        self._text_detector = text_detector
        self._aggregator = aggregator
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def analyze(self, image: Image.Image) -> AnalysisResult:
        """Запустить полный пайплайн анализа и вернуть результат."""
        async with self._semaphore:
            start = time.perf_counter()

            symbol_task = asyncio.to_thread(self._symbol_detector.detect, image)
            text_task = asyncio.to_thread(self._text_detector.detect, image)

            clip_result: ClipInferenceResult
            ocr_result: OcrInferenceResult
            clip_result, ocr_result = await asyncio.gather(symbol_task, text_task)

            result = self._aggregator.aggregate(clip_result.detections, ocr_result.detections)
            result.processing_time_ms = round((time.perf_counter() - start) * 1000, 1)

            logger.info(
                "analysis_complete",
                verdict=result.verdict.value,
                score=result.score,
                clip_best_label=clip_result.best_label,
                clip_best_score=clip_result.best_score,
                ocr_text_length=ocr_result.full_text_length,
                processing_time_ms=result.processing_time_ms,
            )
            return result

    @property
    def models_loaded(self) -> bool:
        """Проверить, загружены ли все ML-модели."""
        return self._symbol_detector.is_loaded and self._text_detector.is_loaded
