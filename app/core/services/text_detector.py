"""TextDetector — извлечение текста через EasyOCR и поиск ключевых слов."""

from __future__ import annotations

from typing import Any

import easyocr
import numpy as np
import torch
from PIL import Image

from app.core.logging import get_logger
from app.core.models import TextDetection
from app.core.models.dto import OcrInferenceResult
from app.core.pipeline.prompts import OCR_KEYWORDS

logger = get_logger(__name__)

_CONTEXT_WINDOW: int = 60


class TextDetector:
    """Извлечение текста из изображений через EasyOCR и поиск ключевых слов."""

    def __init__(self, languages: list[str] | None = None) -> None:
        self._languages = languages or ["en", "ru"]
        self._reader: Any = None

    def load(self) -> None:
        """Инициализировать EasyOCR Reader (при первом запуске загружает веса модели)."""
        logger.info("loading_easyocr", languages=self._languages)
        self._reader = easyocr.Reader(self._languages, gpu=torch.cuda.is_available())
        logger.info("easyocr_loaded")

    @property
    def is_loaded(self) -> bool:
        """Проверить, загружен ли ридер."""
        return self._reader is not None

    def detect(self, image: Image.Image) -> OcrInferenceResult:
        """Извлечь текст из изображения и сопоставить с ключевыми словами."""
        if not self.is_loaded:
            raise RuntimeError("TextDetector не загружен. Вызовите load().")

        img_array: np.ndarray = np.array(image)
        raw_results: list[Any] = self._reader.readtext(img_array, detail=1)

        full_text = " ".join(entry[1] for entry in raw_results if entry[1].strip())
        full_text_lower = full_text.lower()

        if not full_text_lower.strip():
            logger.debug("ocr_no_text_found")
            return OcrInferenceResult(detections=[], full_text_length=0)

        logger.debug("ocr_extracted_text", length=len(full_text))

        detections: list[TextDetection] = []
        seen_keywords: set[str] = set()

        for canonical, variants in OCR_KEYWORDS.items():
            for variant in variants:
                variant_lower = variant.lower()
                if variant_lower in full_text_lower and canonical not in seen_keywords:
                    seen_keywords.add(canonical)
                    context = self._extract_context(full_text_lower, variant_lower)
                    conf = self._estimate_confidence(raw_results, variant_lower)
                    detections.append(
                        TextDetection(
                            keyword=canonical,
                            context=context,
                            confidence=round(conf, 4),
                        )
                    )
                    break

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("text_detection_done", detections=len(detections))
        return OcrInferenceResult(detections=detections, full_text_length=len(full_text))

    @staticmethod
    def _extract_context(text: str, keyword: str) -> str:
        """Вернуть фрагмент текста вокруг первого вхождения ключевого слова."""
        idx = text.find(keyword)
        if idx == -1:
            return ""
        start = max(0, idx - _CONTEXT_WINDOW // 2)
        end = min(len(text), idx + len(keyword) + _CONTEXT_WINDOW // 2)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        return snippet

    @staticmethod
    def _estimate_confidence(ocr_entries: list[Any], keyword_lower: str) -> float:
        """Выбрать наивысшую OCR-уверенность среди записей, содержащих ключевое слово."""
        best: float = 0.0
        for entry in ocr_entries:
            text_lower = entry[1].lower()
            conf: float = float(entry[2]) if len(entry) > 2 else 0.5
            if keyword_lower in text_lower and conf > best:
                best = conf
        return best if best > 0.0 else 0.5
