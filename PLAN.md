# PLAN.md — ревизия проекта PulseNet / ТРИПЛЕКСИЯ

Проверено: `pytest` → 298 passed, 0 skipped (нужен пакет `mcp`, он в requirements.txt).
Веб-интерфейс, docker и `run_server.bat` в песочнице не запускались (Windows) — проверить вручную.

## 1. Карта: что за что отвечает

Точки входа
- `python -m pulsenet serve` — веб-чат и HTTP API (порт 8000). Логика: `app.py`, HTTP: `server.py`.
- `python -m pulsenet.unified_mcp serve` — единственный MCP-сервер памяти и инструментов (порт 8010, путь `/mcp`).
- `run_server.bat` — запуск docker (nginx, searxng), EasyDiffusion, MCP и веба.
- `python -m pulsenet ingest|consolidate|train-sleep` — CLI.

Ядро чата
- `app.py` — оркестрация чата. `mcp_client.py` — клиент к MCP-серверу памяти (`RemoteMemory`).
- `llm.py` — клиент llama.cpp / LM Studio. `toolcalls.py`, `tools.py` — агентный цикл и инструменты модели.
- `memory_service.py` — сборка эмбеддера и экстрактора для памяти.
- `chat_store.py`, `settings_store.py`, `workspace.py`, `tts.py`.

Память ТРИПЛЕКСИЯ (`triplex/`)
- `journal.py` — append-only журнал фактов с инвариантами I1–I10; `fact_to_dict` — сериализация для MCP.
- `docstore.py`, `index.py` (LSH + вектор + RRF), `memory.py` (трёхэтапный recall, конфликты, забывание).
- `extract.py` — извлечение фактов LLM с проверкой evidence. `encoder.py`, `embed_server.py` — эмбеддинги.
- `sleep.py` — «сон» (M6): отбор фактов и LoRA-обучение, нужен GPU.

Единый MCP (`unified_mcp/`)
- `server.py` — все инструменты: triplex_*, identity_*, core_*, web_*, image_generate, code_*, whoami, pulsenet_guide.
- `auth.py` — токены, `store.py` — хранилище по пользователям, `web.py` — SearXNG, `image.py` — EasyDiffusion, `code.py` — песочница, `activity.py` — журнал действий.

Вход и почта: `auth_routes.py`, `auth_store.py`, `auth_session.py`, `auth_config.py`, `mailer.py`, `envfile.py`.
Фронтенд: `web/`. Тесты: `tests/`. Исследовательские проверки качества: `eval/`.
Документация: `README.md`, `ROADMAP.md`, `AGENTS.md`, `docs/`, `.opencode/skills/triplex-mcp/SKILL.md`.

## 2. Сделано

### 2.1 Объединение MCP-слоя
Выяснилось, что «два живых сервера» — неверная картина: веб-сервер `/mcp` не обслуживает, а `triplex/mcp.py` (JSON-RPC-сервер `pulsenet-triplex`) никем не запускался. Чат ходит только в `pulsenet-unified` (`PULSENET_MCP_URL`, по умолчанию 127.0.0.1:8010/mcp, токен `PULSENET_CHAT_MCP_TOKEN`).
- Удалён `triplex/mcp.py` (класс `TriplexMCP`, JSON-RPC-протокол, `parse_body`) — мёртвый код.
- Из него нужна была одна функция: `_fact_dict` перенесена в `triplex/journal.py` как `fact_to_dict`.
- В unified добавлен недостающий инструмент `triplex_consolidate` (сухой прогон «сна»).
- Удалён `tests/test_triplex_mcp.py`: тестировал протокол мёртвого класса. Инварианты журнала проверяются в `test_triplex_journal.py`, сквозной путь ingest→recall — в `test_triplex_memory.py`, сквозной HTTP с MCP-клиентом — в `test_unified_mcp.py`.
- Сквозной тест unified теперь проверяет `triplex_consolidate` и `triplex_check`.
- В `triplex/__init__.py` убраны экспорты `TriplexMCP` и `parse_body`.
- Баннер HTTP-сервера: `pulsenet-triplex/2.2` → `pulsenet/2.2`.

Итог: 274 → 298 тестов (после включения пакета `mcp` все тесты unified выполняются, раньше 3 модуля пропускались). Количество инструментов и их поведение не изменились, кроме добавленного `triplex_consolidate`.

### 2.2 Исправлено ранее
- `omegaconf` удалён из requirements.txt (не импортируется).
- `.env.example`: пустой `PULSENET_MCP_TOKEN=` дублировался и затирал токен из строки выше. Дубль удалён. Дубль `TRIPLEX_EXTRACT` удалён.
- `docker-compose.yml`: `./static` → `./web/static`.
- `make-archive.ps1` не упаковывает `logs/`, `data/`, `data_unified/`, `.pytest_cache/`.

## 2.3 Документация (очистка)
- README переписан: только запуск, состав и ссылки. Устаревшие утверждения убраны: `mcp.py` в составе, `/mcp` на веб-сервере, `TRIPLEX_MCP_TOKEN`, `setup_env.bat`, `stop_server.bat`, «`/api` открыт всем» (на самом деле защищён входом).
- `docs/MEMORY.md` — устройство памяти (из README).
- `docs/MCP.md` — единая справка по MCP: настройка, токены, клиенты, 29 инструментов (сверено с запущенным сервером), ошибки. Заменяет `UNIFIED_MCP.md` и `OPENCODE.md`.
- `AGENTS.md` — только правила работы агента. Утверждения о BM25, cross-encoder, simhash и query expansion удалены: их нет в коде, это описание другого проекта.
- `SKILL.md` сокращён и ссылается на `docs/MCP.md`. Блок про `tools.browser.*` удалён: это инструменты другой среды, не этого MCP.
- `ROADMAP.md` — только открытые задачи; готовые пункты, таблица «Help wanted», хронология удалены.
- `.env.example`: `PULSENET_MCP_HOSTS` (код его не читает) заменён на `MCP_ALLOWED_HOSTS`; добавлены `MCP_OWNER_ID`, `MCP_ADMIN_USERS`, `CODE_ACCESS_USERS`, `CODE_ROOTS`.
- `run_server.bat`: сообщение об отсутствии `.venv` теперь указывает команды вместо несуществующего `setup_env.bat`.
- `docs/CHAT_AGENTS.example.md` оставлен: правила для агента внутри приложения, а не для opencode.
- `docs/AGENT_WORKSPACE.md` оставлен без изменений. `docs/TTS_SILERO.md` заменён на `docs/TTS.md`, который описывает оба движка.

## 2.4 Лишнее, найдено во второй проверке
- Удалены 7 PNG-иконок в `web/static/icons/` — ни одна не используется.
- Удалена функция `_health` в `server.py` — мёртвый код, `/health` обрабатывается выше по коду.
- В ROADMAP исправлена ссылка на `eval/fixtures/queries.jsonl`: файла нет, теперь написано «создать».

## 2.5 Озвучка: Piper вместо XTTS
- XTTS-v2 удалён: в установке требовались torch, torchaudio, torchcodec и FFmpeg, и это не работало на машине пользователя.
- Добавлен движок Piper (`pulsenet/tts_piper.py`): только `piper-tts` и onnxruntime, без torch и FFmpeg. Голоса — файлы `.onnx`, по одному на язык (`PIPER_VOICE_RU`, `PIPER_VOICE_EN`).
- Движок по умолчанию: `TTS_ENGINE=piper`. Silero остаётся как `TTS_ENGINE=silero` и требует torch.
- `detect_language` перенесён в `pulsenet/tts.py` и используется обоими движками.
- Тесты: `tests/test_tts_piper.py` (заглушка голоса). Всего 313 тестов.
- Не проверено в песочнице: загрузка голосов с huggingface.co и звук на реальной машине.

## 3. Решения, которые нужны от тебя (в коде не трогал)
1. `migrate_memory.py` — одноразовый перенос `data/triplex` → `data_unified`. Если перенос сделан, удалить модуль и связанные тесты в `test_remote_memory.py`.
2. `triplex/embed_server.py` — отдельный сервер эмбеддингов. Если LM Studio или llama.cpp отдаёт `/v1/embeddings`, можно убрать.
3. `triplex/sleep.py` (M6) — нужен GPU, в рантайме не используется. Оставить, но вынести зависимости (torch, peft) в отдельный файл.
4. Крупные файлы `app.py` (1541), `server.py` (1012), `static/app.js` (4238) — кандидаты на разбиение по доменам, не на удаление.
5. `eval/` оставить вне рантайма.
6. `web/sw.js` — заглушка: снимает регистрацию старого service worker. Удалить можно, когда все браузеры получили её; до тех пор оставить.

## 4. Следующие шаги
1. Решить пункты 1–3 раздела 3.
2. Ручная проверка на Windows: `run_server.bat`, веб-чат, вызов инструментов памяти из opencode, `triplex_consolidate`.
3. Разбиение `app.py` и `app.js` по доменам — по одному модулю за шаг, после каждого шага тесты.

Тесты: `pip install -r requirements-dev.txt` → `python -m pytest`.
