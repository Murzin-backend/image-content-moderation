# image-content-moderation

Сервис анализа изображений для модерации контента. Определяет наличие ЛГБТ-символики, флагов, эмблем, баннеров, плакатов, стикеров, поведенческих паттернов (однополые пары, прайд-парады, дрэг-шоу) и текстовых маркеров на изображениях через OCR. Возвращает объяснимые результаты (метки и вероятности) через REST API.

---

## Архитектура

```
Client (multipart/form-data image)
  │
  ▼
FastAPI Router
  ├── POST /api/analyze    — анализ изображения
  └── GET  /api/health     — health check
  │
  ▼
AnalysisService (оркестратор, asyncio.Semaphore для ограничения нагрузки)
  ├── SymbolDetector   — CLIP zero-shot классификация (ensemble mean pooling)
  ├── TextDetector     — EasyOCR + keyword matching
  └── ResultAggregator — агрегация score / verdict
```

### ML Pipeline

| Компонент | Модель | Назначение |
|-----------|--------|------------|
| **SymbolDetector** | OpenAI CLIP ViT-B/32 | Zero-shot бинарная классификация: каждый позитивный label (ensemble из 2–3 перефразировок) vs все негативные промпты через softmax |
| **TextDetector** | EasyOCR (EN + RU) | Извлечение текста с изображений + поиск ключевых слов (pride, лгбт, queer и др.) |
| **ResultAggregator** | — | Объединение score из детекторов, вычисление итогового verdict |

### Что распознаётся

**Символика (CLIP):**
- Радужные флаги (6-полосный, прогресс, трансгендерный, бисексуальный, небинарный)
- Баннеры, эмблемы, стикеры, муралы с радужной символикой
- Прайд-парады и шествия
- Однополые пары (мужские и женские)
- Дрэг-перформансы

**Текст (OCR):**
- Ключевые слова на английском и русском: pride, lgbt, queer, gay, lesbian, прайд, лгбт, гей и др.

### Verdict

| Verdict | Score | Описание |
|---------|-------|----------|
| `REJECTED` | ≥ 0.55 | Обнаружена символика или маркеры с высокой уверенностью |
| `SUSPICIOUS` | ≥ 0.35 | Возможно наличие символики, требуется ручная проверка |
| `CLEAN` | < 0.35 | Символика и маркеры не обнаружены |

---

## Быстрый старт

### Предварительные требования

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Mac / Windows / Linux)
- Для GPU-режима: Linux + NVIDIA GPU + драйверы + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Важно: настройка памяти Docker Desktop (Mac / Windows)

Модели CLIP + EasyOCR + PyTorch требуют **минимум 4 ГБ RAM**. По умолчанию Docker Desktop выделяет ~2 ГБ.

**Mac:**
1. Docker Desktop → Settings → Resources
2. Memory → установить **6 ГБ** (рекомендуется 8 ГБ)
3. Apply & Restart

**Windows:**
1. Docker Desktop → Settings → Resources → Advanced
2. Memory → установить **6 ГБ**
3. Apply & Restart

### Запуск в Docker (CPU-режим, Mac / Windows / Linux)

```bash
# 1. Клонировать репозиторий
git clone <repo-url>
cd image-content-moderation

# 2. Создать .env файл
cp .env.example .env

# 3. Собрать и запустить
docker compose up --build

# Первый запуск занимает ~5-10 минут (скачивание моделей CLIP ~340 МБ и EasyOCR ~100 МБ).
# Модели кешируются в Docker volumes и повторно не скачиваются.
```

Сервис будет доступен по адресу: `http://localhost:8000`

### Запуск в Docker (GPU-режим, только Linux + NVIDIA)

```bash
# 1. Создать .env файл
cp .env.example .env

# 2. Запустить GPU-профиль
docker compose --profile gpu up --build

# GPU-профиль использует nvidia/cuda базовый образ и CLIP_DEVICE=cuda.
# Требуется установленный nvidia-container-toolkit.
```

### Запуск локально (без Docker)

```bash
# 1. Установить Poetry (если ещё не установлен)
curl -sSL https://install.python-poetry.org | python3 -

# 2. Указать Poetry использовать нужную версию Python
poetry env use python3.11

# 3. Установить все зависимости
poetry install

# 4. Создать .env файл
cp .env.example .env

# 5. Запустить сервис
poetry run uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

### Настройка интерпретатора Python из Poetry в PyCharm

```bash
# Узнать путь к виртуальному окружению Poetry
poetry env info --path
# Пример: /Users/user/Library/Caches/pypoetry/virtualenvs/image-content-moderation-AbCdEf-py3.11
```

В **PyCharm**:
1. `Settings` → `Project` → `Python Interpreter`
2. Шестерёнка → `Add Interpreter` → `Add Local Interpreter`
3. Выбрать `Poetry Environment` → `Existing environment`
4. Указать путь: `<вывод poetry env info --path>/bin/python`

---

## Использование API

### POST /api/analyze

Анализ изображения. Принимает `multipart/form-data` с полем `file`.

**Поддерживаемые форматы:** JPEG, PNG, WebP, BMP, TIFF
(тип определяется автоматически по содержимому файла, даже если curl отправляет `application/octet-stream`)

```bash
# Анализ изображения
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@image.jpg"

# С красивым выводом (нужен jq: brew install jq)
curl -s -X POST http://localhost:8000/api/analyze \
  -F "file=@image.jpg" | jq
```

**Пример ответа:**
```json
{
  "verdict": "REJECTED",
  "score": 0.9959,
  "symbol_detections": [
    {"label": "rainbow flag", "confidence": 0.9959},
    {"label": "pride flag", "confidence": 0.9492}
  ],
  "text_detections": [
    {"keyword": "pride", "context": "...happy pride month...", "confidence": 0.95}
  ],
  "processing_time_ms": 355
}
```

### GET /api/health

```bash
curl http://localhost:8000/api/health
```

```json
{"status": "healthy", "models_loaded": true}
```

### Коды ошибок

| Код | Описание |
|-----|----------|
| 413 | Файл слишком большой (> MAX_IMAGE_SIZE_MB) |
| 415 | Неподдерживаемый формат файла |
| 422 | Не удалось декодировать изображение |
| 503 | ML-модели ещё не загружены |

---

## Конфигурация

Все настройки задаются через переменные окружения (файл `.env`):

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `APP_NAME` | `image-content-moderation` | Имя сервиса |
| `DEBUG` | `false` | Режим отладки (включает `/docs` и `/redoc`) |
| `LOG_LEVEL` | `INFO` | Уровень логирования |
| `HOST` | `0.0.0.0` | Хост |
| `PORT` | `8000` | Порт |
| `CLIP_MODEL_NAME` | `ViT-B/32` | Модель CLIP |
| `CLIP_DEVICE` | auto | Устройство (`cuda` / `cpu`, авто-определение) |
| `EASYOCR_LANGUAGES` | `en,ru` | Языки OCR через запятую |
| `CONFIDENCE_THRESHOLD` | `0.55` | Порог для REJECTED |
| `SUSPICIOUS_THRESHOLD` | `0.35` | Порог для SUSPICIOUS |
| `MAX_IMAGE_SIZE_MB` | `10` | Макс. размер файла (МБ) |
| `MAX_CONCURRENT_ANALYSES` | `4` | Макс. одновременных инференсов (семафор) |

---

## Структура проекта

```
app/
├── __init__.py
├── exceptions.py              # Базовые классы исключений (AppException, PayloadTooLargeError, ...)
├── main.py                    # Фабрика приложения, lifespan (загрузка моделей)
├── api/
│   ├── router.py              # Центральный маршрутизатор (/api)
│   ├── exceptions.py          # Обработчики исключений FastAPI, collect_responses()
│   ├── controllers/
│   │   ├── analyze.py         # POST /api/analyze
│   │   └── health.py          # GET /api/health
│   └── serializers/
│       └── responses.py       # Pydantic-схемы ответов (AnalysisResponse, ErrorResponse, ...)
└── core/
    ├── config.py              # Settings (pydantic-settings, env)
    ├── logging.py             # structlog настройка
    ├── models/
    │   ├── domain.py          # Dataclass-модели (Verdict, SymbolDetection, AnalysisResult, ...)
    │   └── dto.py             # DTO для ML-инференса (ClipInferenceResult, OcrInferenceResult)
    ├── pipeline/
    │   ├── aggregator.py      # ResultAggregator — score → verdict
    │   └── prompts.py         # CLIP промпты (ensemble PromptGroup) + OCR ключевые слова
    ├── services/
    │   ├── analysis.py        # AnalysisService — оркестратор (asyncio.to_thread + Semaphore)
    │   ├── symbol_detector.py # SymbolDetector — CLIP zero-shot + ensemble mean pooling
    │   └── text_detector.py   # TextDetector — EasyOCR + keyword matching
    └── utils/
        └── image.py           # Валидация, ресайз, определение MIME по magic bytes
```

---

## Производительность и масштабирование

### Текущая архитектура

- **asyncio.to_thread()** — ML-инференс (CPU/GPU-bound) выполняется в пуле потоков, event loop не блокируется
- **asyncio.gather()** — CLIP и EasyOCR работают параллельно
- **asyncio.Semaphore** — ограничивает одновременные инференсы (защита от OOM)
- Изображения автоматически ресайзятся до 1024px

### Горизонтальное масштабирование

Для высокой нагрузки — запуск нескольких реплик:

```bash
docker compose up --build --scale app=3
```

При этом нужно убрать `container_name` из `docker-compose.yml` и добавить reverse-proxy (nginx / traefik).

### Когда нужна очередь задач (Celery / Redis)?

- При ответе > 5 секунд и необходимости асинхронной обработки (webhook callback)
- При обработке батчей изображений
- Для текущего MVP (ответ ~0.5–1.5 сек) — **не нужна**

---

## Тестирование

```bash
poetry run pytest
```

---

## Стек

- Python 3.9+ / Poetry
- FastAPI + Pydantic v2 + pydantic-settings
- PyTorch + OpenAI CLIP (zero-shot classification)
- EasyOCR (OCR EN + RU)
- structlog (structured logging)
- Docker + Docker Compose (CPU + GPU profiles)
