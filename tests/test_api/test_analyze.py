"""Тесты для POST /api/analyze."""

from __future__ import annotations

import io

import pytest
from httpx import AsyncClient
from PIL import Image


def _create_test_image(fmt: str = "JPEG") -> bytes:
    """Создать маленькое валидное изображение в памяти."""
    img = Image.new("RGB", (64, 64), color="white")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.mark.asyncio
async def test_analyze_returns_200_for_valid_image(client: AsyncClient) -> None:
    data = _create_test_image()
    response = await client.post(
        "/api/analyze",
        files={"file": ("test.jpg", data, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "CLEAN"
    assert "score" in body
    assert "processing_time_ms" in body


@pytest.mark.asyncio
async def test_analyze_rejects_unsupported_media_type(client: AsyncClient) -> None:
    response = await client.post(
        "/api/analyze",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_analyze_rejects_oversized_file(client: AsyncClient) -> None:
    big_data = b"\x00" * (11 * 1024 * 1024)
    response = await client.post(
        "/api/analyze",
        files={"file": ("big.jpg", big_data, "image/jpeg")},
    )
    assert response.status_code == 413


@pytest.mark.asyncio
async def test_analyze_png(client: AsyncClient) -> None:
    data = _create_test_image(fmt="PNG")
    response = await client.post(
        "/api/analyze",
        files={"file": ("test.png", data, "image/png")},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    response = await client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["models_loaded"] is True
