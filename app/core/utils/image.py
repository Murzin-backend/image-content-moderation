"""Утилиты для валидации и обработки изображений."""

from __future__ import annotations

import io
import mimetypes

from PIL import Image

from app.core.logging import get_logger
from app.exceptions import ImageDecodeError, PayloadTooLargeError, UnsupportedMediaTypeError

logger = get_logger(__name__)

_MAGIC_SIGNATURES: list[tuple[bytes, int, str]] = [
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"RIFF", 0, "image/webp"),
    (b"BM", 0, "image/bmp"),
    (b"II\x2a\x00", 0, "image/tiff"),
    (b"MM\x00\x2a", 0, "image/tiff"),
]


def resolve_content_type(content_type: str | None, data: bytes, filename: str | None = None) -> str:
    """Определить реальный MIME-тип файла.

    Приоритет: content_type от клиента → magic bytes → расширение файла.
    """
    if content_type and content_type.startswith("image/"):
        return content_type

    for signature, offset, mime in _MAGIC_SIGNATURES:
        if data[offset:offset + len(signature)] == signature:
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            logger.debug("resolved_mime_by_magic", detected=mime, original=content_type)
            return mime

    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed and guessed.startswith("image/"):
            logger.debug("resolved_mime_by_extension", detected=guessed, original=content_type)
            return guessed

    return content_type or "application/octet-stream"


def validate_content_type(content_type: str | None, allowed: list[str]) -> None:
    """Проверить, что тип содержимого входит в список допустимых."""
    if not content_type or content_type not in allowed:
        raise UnsupportedMediaTypeError(
            f"Неподдерживаемый формат: {content_type!r}. "
            f"Допустимые: {', '.join(allowed)}"
        )


def validate_file_size(data: bytes, max_bytes: int) -> None:
    """Проверить, что размер файла не превышает лимит."""
    if len(data) > max_bytes:
        mb = max_bytes / (1024 * 1024)
        raise PayloadTooLargeError(
            f"Файл слишком большой ({len(data) / (1024 * 1024):.1f} МБ). "
            f"Максимум: {mb:.0f} МБ"
        )


_MAX_DIMENSION: int = 1024


def load_image(data: bytes) -> Image.Image:
    """Открыть байты как PIL Image, конвертировать в RGB и ограничить размер."""
    try:
        img = Image.open(io.BytesIO(data))
        img = img.convert("RGB")
        img = _resize_if_needed(img, _MAX_DIMENSION)
        return img
    except Exception as exc:
        logger.warning("image_load_failed", error=str(exc))
        raise ImageDecodeError(f"Не удалось декодировать изображение: {exc}") from exc


def _resize_if_needed(img: Image.Image, max_dim: int) -> Image.Image:
    """Уменьшить изображение, если любая сторона превышает max_dim."""
    w, h = img.size
    if w <= max_dim and h <= max_dim:
        return img

    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.debug("resizing_image", original=f"{w}x{h}", resized=f"{new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)

