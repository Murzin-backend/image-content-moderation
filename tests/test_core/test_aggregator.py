"""Тесты для ResultAggregator."""

from __future__ import annotations

import pytest

from app.core.models import SymbolDetection, TextDetection, Verdict
from app.core.pipeline.aggregator import ResultAggregator


@pytest.fixture()
def aggregator() -> ResultAggregator:
    return ResultAggregator(
        confidence_threshold=0.55,
        suspicious_threshold=0.35,
    )


class TestResultAggregator:
    def test_clean_when_no_detections(self, aggregator: ResultAggregator) -> None:
        result = aggregator.aggregate([], [])
        assert result.verdict == Verdict.CLEAN
        assert result.score == 0.0

    def test_rejected_on_high_symbol_confidence(self, aggregator: ResultAggregator) -> None:
        symbols = [SymbolDetection(label="rainbow flag", confidence=0.8)]
        result = aggregator.aggregate(symbols, [])
        assert result.verdict == Verdict.REJECTED
        assert result.score >= 0.55

    def test_suspicious_on_medium_confidence(self, aggregator: ResultAggregator) -> None:
        symbols = [SymbolDetection(label="rainbow emblem", confidence=0.40)]
        result = aggregator.aggregate(symbols, [])
        assert result.verdict == Verdict.SUSPICIOUS

    def test_rejected_on_high_text_confidence(self, aggregator: ResultAggregator) -> None:
        texts = [TextDetection(keyword="pride", context="happy pride", confidence=0.95)]
        result = aggregator.aggregate([], texts)
        assert result.verdict == Verdict.REJECTED

    def test_text_weight_reduces_score(self, aggregator: ResultAggregator) -> None:
        # Text confidence 0.60 * weight 0.85 = 0.51 → should be SUSPICIOUS, not REJECTED
        texts = [TextDetection(keyword="lgbt", context="lgbt rights", confidence=0.60)]
        result = aggregator.aggregate([], texts)
        assert result.verdict == Verdict.SUSPICIOUS

    def test_combined_detections(self, aggregator: ResultAggregator) -> None:
        symbols = [SymbolDetection(label="rainbow flag", confidence=0.70)]
        texts = [TextDetection(keyword="pride", context="pride march", confidence=0.90)]
        result = aggregator.aggregate(symbols, texts)
        assert result.verdict == Verdict.REJECTED
        # Score should be max of symbol (0.70) and text (0.90*0.85=0.765) → 0.765
        assert result.score == pytest.approx(0.765, abs=0.01)

    def test_score_uses_max_of_symbol_and_weighted_text(
        self, aggregator: ResultAggregator
    ) -> None:
        symbols = [SymbolDetection(label="flag", confidence=0.30)]
        texts = [TextDetection(keyword="rainbow", context="rainbow", confidence=0.30)]
        result = aggregator.aggregate(symbols, texts)
        # max(0.30, 0.30*0.85=0.255) → 0.30
        assert result.score == pytest.approx(0.30, abs=0.01)
        assert result.verdict == Verdict.CLEAN

    def test_detections_are_preserved_in_result(self, aggregator: ResultAggregator) -> None:
        symbols = [SymbolDetection(label="pride flag", confidence=0.60)]
        texts = [TextDetection(keyword="queer", context="queer art", confidence=0.80)]
        result = aggregator.aggregate(symbols, texts)
        assert len(result.symbol_detections) == 1
        assert len(result.text_detections) == 1
        assert result.symbol_detections[0].label == "pride flag"
        assert result.text_detections[0].keyword == "queer"
