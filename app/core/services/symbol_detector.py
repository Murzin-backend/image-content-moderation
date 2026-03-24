"""SymbolDetector — обнаружение символики через CLIP zero-shot классификацию."""

from __future__ import annotations

import os
from typing import Any

import clip
import torch
from PIL import Image

from app.core.logging import get_logger
from app.core.models import SymbolDetection
from app.core.models.dto import ClipInferenceResult
from app.core.pipeline.prompts import POSITIVE_GROUPS, NEGATIVE_GROUPS

logger = get_logger(__name__)


class SymbolDetector:
    """Обнаружение символов, флагов и эмблем через OpenAI CLIP.

    Ensemble-подход: для каждого label несколько промптов усредняются
    (mean pooling) в один вектор. Каждый позитивный label сравнивается
    со всеми негативными через softmax.
    """

    _MIN_REPORT_CONFIDENCE: float = 0.10

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu", cache_dir: str | None = None) -> None:
        self._device = device
        self._model_name = model_name
        self._cache_dir = cache_dir or os.environ.get("CLIP_CACHE_DIR", os.path.expanduser("~/.cache/clip"))
        self._model: Any = None
        self._preprocess: Any = None
        self._pos_features: torch.Tensor | None = None
        self._neg_features: torch.Tensor | None = None
        self._positive_labels: list[str] = []

    def load(self) -> None:
        """Загрузить модель CLIP и закодировать промпты с ensemble mean pooling."""
        logger.info("loading_clip", model=self._model_name, device=self._device, cache_dir=self._cache_dir)
        self._model, self._preprocess = clip.load(
            self._model_name, device=self._device, download_root=self._cache_dir,
        )

        self._positive_labels = [g.label for g in POSITIVE_GROUPS]
        self._pos_features = self._encode_groups(POSITIVE_GROUPS)
        self._neg_features = self._encode_groups(NEGATIVE_GROUPS)

        logger.info(
            "clip_loaded",
            positive_groups=len(POSITIVE_GROUPS),
            negative_groups=len(NEGATIVE_GROUPS),
        )

    def _encode_groups(self, groups: list) -> torch.Tensor:
        """Закодировать группы промптов: mean pooling по каждой группе."""
        group_features: list[torch.Tensor] = []

        for group in groups:
            tokens = clip.tokenize(group.prompts).to(self._device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                mean_feature = features.mean(dim=0)
                mean_feature = mean_feature / mean_feature.norm()
                group_features.append(mean_feature)

        return torch.stack(group_features)

    @property
    def is_loaded(self) -> bool:
        """Проверить, загружена ли модель."""
        return self._model is not None

    def detect(self, image: Image.Image) -> ClipInferenceResult:
        """Выполнить zero-shot классификацию изображения.

        Для каждого позитивного label:
        1. Собрать logits = [sim(image, pos_i), sim(image, neg_1), ..., sim(image, neg_N)]
        2. Softmax → вероятность pos_i
        Вернуть ClipInferenceResult с обнаружениями.
        """
        if not self.is_loaded:
            raise RuntimeError("Модель SymbolDetector не загружена. Вызовите load().")

        image_input = self._preprocess(image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            pos_sims = (image_features @ self._pos_features.T).squeeze(0)
            neg_sims = (image_features @ self._neg_features.T).squeeze(0)

        results: list[SymbolDetection] = []
        raw_pos = [round(s.item(), 4) for s in pos_sims]
        neg_max = round(neg_sims.max().item(), 4)

        for i in range(len(POSITIVE_GROUPS)):
            logits = torch.cat([pos_sims[i:i+1], neg_sims]) * 100.0
            prob = logits.softmax(dim=0)[0].item()

            if prob >= self._MIN_REPORT_CONFIDENCE:
                results.append(SymbolDetection(
                    label=self._positive_labels[i],
                    confidence=round(prob, 4),
                ))

        results.sort(key=lambda d: d.confidence, reverse=True)

        best_score = results[0].confidence if results else 0.0
        best_label = results[0].label if results else None

        logger.info(
            "symbol_detection_done",
            detections=len(results),
            best_score=round(best_score, 4),
            best_label=best_label,
        )

        return ClipInferenceResult(
            detections=results,
            best_score=best_score,
            best_label=best_label,
            positive_similarities=raw_pos,
            negative_max=neg_max,
        )
