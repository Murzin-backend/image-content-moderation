"""Тесты для TextDetector и словарей ключевых слов."""

from __future__ import annotations

import pytest

from app.core.pipeline.prompts import OCR_KEYWORDS, get_all_keyword_variants
from app.core.services.text_detector import TextDetector


class TestOCRKeywords:
    def test_all_keywords_are_lowercase(self) -> None:
        for variant in get_all_keyword_variants():
            assert variant == variant.lower(), f"Вариант {variant!r} не в нижнем регистре"

    def test_known_keywords_present(self) -> None:
        variants = get_all_keyword_variants()
        for expected in ("pride", "lgbt", "rainbow", "прайд", "лгбт"):
            assert expected in variants, f"Отсутствует ключевое слово: {expected}"

    def test_canonical_keys_have_variants(self) -> None:
        for canonical, variants in OCR_KEYWORDS.items():
            assert len(variants) > 0, f"Ключевое слово {canonical!r} не имеет вариантов"

    def test_context_extraction_static_method(self) -> None:
        text = "this is a long text about a pride march in the city center last summer"
        ctx = TextDetector._extract_context(text, "pride")
        assert "pride" in ctx
        assert len(ctx) <= 80
