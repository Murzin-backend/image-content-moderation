"""Настройки приложения, загружаемые из переменных окружения."""

from __future__ import annotations

import torch
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Центральная конфигурация — каждое поле соответствует переменной окружения."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    APP_NAME: str = "image-content-moderation"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    CLIP_MODEL_NAME: str = "ViT-B/32"
    CLIP_DEVICE: str = ""

    EASYOCR_LANGUAGES: str = "en,ru"

    CONFIDENCE_THRESHOLD: float = 0.55
    SUSPICIOUS_THRESHOLD: float = 0.35

    MAX_IMAGE_SIZE_MB: int = 10
    MAX_CONCURRENT_ANALYSES: int = 4
    ALLOWED_CONTENT_TYPES: list[str] = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
    ]

    @property
    def max_image_bytes(self) -> int:
        """Максимальный размер изображения в байтах."""
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024

    @property
    def clip_device_resolved(self) -> str:
        """Определить устройство для CLIP (cuda или cpu)."""
        if self.CLIP_DEVICE:
            return self.CLIP_DEVICE
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def easyocr_language_list(self) -> list[str]:
        """Список языков для EasyOCR."""
        return [lang.strip() for lang in self.EASYOCR_LANGUAGES.split(",")]

    @field_validator("LOG_LEVEL")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()


def get_settings() -> Settings:
    """Получить экземпляр настроек."""
    return Settings()
