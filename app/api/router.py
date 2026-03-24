"""Главный маршрутизатор API — агрегирует все контроллеры."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.controllers.analyze import router as analyze_router
from app.api.controllers.health import router as health_router

api_router = APIRouter(prefix="/api")

api_router.include_router(analyze_router, tags=["Анализ"])
api_router.include_router(health_router, tags=["Здоровье"])

