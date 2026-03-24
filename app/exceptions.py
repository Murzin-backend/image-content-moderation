"""Классы исключений приложения."""

from __future__ import annotations


class AppException(Exception):
    """Базовое исключение приложения с привязкой к HTTP-статусу."""

    status_code: int = 500
    detail: str = "Внутренняя ошибка сервера"
    description: str = "Ошибка сервера"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.__class__.detail
        super().__init__(self.detail)


class PayloadTooLargeError(AppException):
    """Размер загруженного файла превышает допустимый лимит."""

    status_code = 413
    detail = "Размер файла превышает допустимый лимит"
    description = "Файл слишком большой"


class UnsupportedMediaTypeError(AppException):
    """Неподдерживаемый тип медиа-файла."""

    status_code = 415
    detail = "Неподдерживаемый формат файла"
    description = "Неподдерживаемый тип медиа"


class ImageDecodeError(AppException):
    """Ошибка декодирования изображения."""

    status_code = 422
    detail = "Не удалось декодировать изображение"
    description = "Ошибка валидации"


class ModelNotReadyError(AppException):
    """ML-модели ещё не загружены."""

    status_code = 503
    detail = "ML-модели ещё не загружены. Попробуйте позже."
    description = "Модели не загружены"

