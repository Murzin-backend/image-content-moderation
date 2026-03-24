"""Промпты CLIP и словари ключевых слов OCR.

Используется ensemble-подход: для каждого label задаётся несколько перефразировок.
При загрузке модели embeddings группы усредняются (mean pooling) —
стандартный приём для повышения точности CLIP zero-shot.

Ключевые слова OCR сопоставляются без учёта регистра с извлечённым текстом.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptGroup:
    """Группа промптов для одного label (ensemble)."""

    label: str
    prompts: list[str]


POSITIVE_GROUPS: list[PromptGroup] = [
    PromptGroup(
        label="rainbow flag",
        prompts=[
            "a photo of a LGBT pride rainbow flag with red orange yellow green blue and purple stripes",
            "a rainbow flag with six colored horizontal stripes at a festival",
            "a close-up of a rainbow-striped pride flag waving in the wind",
        ],
    ),
    PromptGroup(
        label="pride flag",
        prompts=[
            "a photo of a LGBT pride flag at a parade or march",
            "people waving pride flags with rainbow colors at a demonstration",
        ],
    ),
    PromptGroup(
        label="transgender flag",
        prompts=[
            "a photo of a transgender pride flag with light blue pink and white stripes",
            "a trans pride flag with pastel blue and pink horizontal stripes",
        ],
    ),
    PromptGroup(
        label="bisexual flag",
        prompts=[
            "a photo of a bisexual pride flag with magenta purple and blue stripes",
        ],
    ),
    PromptGroup(
        label="progress pride flag",
        prompts=[
            "a photo of a progress pride flag with chevron triangle and rainbow stripes",
            "a pride flag with an arrow-shaped chevron on the left side",
        ],
    ),
    PromptGroup(
        label="non-binary flag",
        prompts=[
            "a photo of a non-binary pride flag with yellow white purple and black stripes",
        ],
    ),
    PromptGroup(
        label="rainbow banner",
        prompts=[
            "a photo of a pride banner with rainbow colors at a LGBT rally or protest",
            "a rainbow-colored banner or sign at a street demonstration",
        ],
    ),
    PromptGroup(
        label="rainbow emblem",
        prompts=[
            "a photo of a LGBT rainbow-colored emblem logo or badge",
            "a pin or badge with rainbow pride colors",
        ],
    ),
    PromptGroup(
        label="rainbow sticker",
        prompts=[
            "a photo of a LGBT pride sticker with rainbow colors and hearts",
        ],
    ),
    PromptGroup(
        label="rainbow mural",
        prompts=[
            "a photo of a rainbow painted on a wall or pavement as a LGBT pride symbol",
        ],
    ),
    PromptGroup(
        label="pride parade",
        prompts=[
            "a photo of people marching at a LGBT pride parade with flags and signs",
            "a crowd of people celebrating at a gay pride festival or march",
            "a pride parade with colorful costumes and rainbow decorations",
        ],
    ),
    PromptGroup(
        label="same-sex couple (male)",
        prompts=[
            "a photo of two men holding hands romantically as a couple",
            "a photo of two men kissing each other on the lips",
            "a romantic photo of a gay male couple embracing",
        ],
    ),
    PromptGroup(
        label="same-sex couple (female)",
        prompts=[
            "a photo of two women holding hands romantically as a couple",
            "a photo of two women kissing each other on the lips",
            "a romantic photo of a lesbian couple embracing",
        ],
    ),
    PromptGroup(
        label="drag performance",
        prompts=[
            "a photo of a drag queen in elaborate costume and heavy makeup performing on stage",
            "a drag performer in a wig and glamorous outfit",
        ],
    ),
]

NEGATIVE_GROUPS: list[PromptGroup] = [
    PromptGroup(
        label="_russian_flag",
        prompts=[
            "a photo of the Russian flag with white blue and red horizontal stripes",
            "the national flag of Russia with three horizontal stripes",
        ],
    ),
    PromptGroup(
        label="_french_flag",
        prompts=["a photo of the French flag with blue white and red vertical stripes"],
    ),
    PromptGroup(
        label="_german_flag",
        prompts=["a photo of the German flag with black red and gold horizontal stripes"],
    ),
    PromptGroup(
        label="_italian_flag",
        prompts=["a photo of the Italian flag with green white and red vertical stripes"],
    ),
    PromptGroup(
        label="_american_flag",
        prompts=["a photo of the American flag with red and white stripes and stars on blue"],
    ),
    PromptGroup(
        label="_national_flag",
        prompts=[
            "a photo of a national flag of a country with colored stripes",
            "a state flag or government flag on a flagpole",
        ],
    ),
    PromptGroup(
        label="_tricolor",
        prompts=["a photo of a tricolor flag with two or three colored horizontal or vertical stripes"],
    ),
    PromptGroup(
        label="_plain_banner",
        prompts=["a photo of a plain single-colored cloth banner or fabric"],
    ),
    PromptGroup(
        label="_sports_flag",
        prompts=["a photo of a sports team flag or pennant at a stadium"],
    ),
    PromptGroup(
        label="_corporate_logo",
        prompts=["a photo of a corporate logo or brand sign"],
    ),
    PromptGroup(
        label="_natural_rainbow",
        prompts=[
            "a photo of a natural rainbow in the sky after rain",
            "a rainbow arc in the sky over a landscape",
        ],
    ),
    PromptGroup(
        label="_landscape",
        prompts=["a photo of a landscape with mountains fields or forest"],
    ),
    PromptGroup(
        label="_food",
        prompts=["a photo of food on a plate or table"],
    ),
    PromptGroup(
        label="_building",
        prompts=["a photo of a building or architecture"],
    ),
    PromptGroup(
        label="_friends",
        prompts=[
            "a photo of friends greeting each other with a handshake or hug",
            "a group of friends posing together for a photo",
        ],
    ),
    PromptGroup(
        label="_family",
        prompts=["a photo of a family with children"],
    ),
    PromptGroup(
        label="_heterosexual_couple",
        prompts=[
            "a photo of a man and a woman holding hands as a romantic couple",
            "a photo of a man and a woman kissing each other",
        ],
    ),
    PromptGroup(
        label="_people",
        prompts=["a photo of a person or group of people standing or walking"],
    ),
    PromptGroup(
        label="_cosplay",
        prompts=["a photo of a costume party cosplay or halloween costume"],
    ),
    PromptGroup(
        label="_concert",
        prompts=["a photo of a music concert or festival with a crowd"],
    ),
    PromptGroup(
        label="_sunset",
        prompts=["a photo of a colorful sunset or sunrise over the horizon"],
    ),
    PromptGroup(
        label="_animal",
        prompts=["a photo of an animal or pet"],
    ),
    PromptGroup(
        label="_vehicle",
        prompts=["a photo of a car truck or vehicle"],
    ),
    PromptGroup(
        label="_document",
        prompts=["a photo of text on paper or a document"],
    ),
]

POSITIVE_LABELS: set[str] = {g.label for g in POSITIVE_GROUPS}

OCR_KEYWORDS: dict[str, list[str]] = {
    "pride": ["pride", "прайд"],
    "lgbt": ["lgbt", "лгбт", "lgbtq", "лгбтк"],
    "rainbow": ["rainbow"],
    "queer": ["queer", "квир"],
    "trans rights": ["trans rights", "права трансгендеров"],
    "love is love": ["love is love", "любовь есть любовь"],
    "equality": ["equality", "равенство"],
    "gay": ["gay", "гей"],
    "lesbian": ["lesbian", "лесби"],
    "bisexual": ["bisexual", "бисексуал"],
    "non-binary": ["non-binary", "nonbinary", "небинарн"],
    "drag": ["drag queen", "drag show", "дрэг"],
}

_ALL_KEYWORD_VARIANTS: set[str] = set()
for _variants in OCR_KEYWORDS.values():
    for _v in _variants:
        _ALL_KEYWORD_VARIANTS.add(_v.lower())


def get_all_keyword_variants() -> set[str]:
    """Вернуть множество всех вариантов ключевых слов в нижнем регистре."""
    return _ALL_KEYWORD_VARIANTS
