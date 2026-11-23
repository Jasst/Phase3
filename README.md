# PulseNet · ТРИПЛЕКСИЯ

Локальный чат с LLM (llama.cpp или LM Studio) и долговременной памятью. Память хранит факты
с источниками и документы; в промпт попадают только релевантные фрагменты. Память доступна
внешним агентам (opencode и другим MCP-клиентам) через единый MCP-сервер. Всё работает локально,
один владелец.

## Статус

Alpha. Около 300 тестов (`pytest`). Работает основной путь: чат, запись и поиск в памяти, MCP.
Не проверено: обучение «Сна» (LoRA) на GPU и качество поиска на реальных русских данных.
Открытые задачи — в [ROADMAP.md](ROADMAP.md).

## Состав

    pulsenet/
      app.py            оркестрация чата: LLM, инструменты, память
      server.py         HTTP: веб-интерфейс, /api/*, авторизация
      llm.py            бэкенды llama.cpp и LM Studio
      mcp_client.py     клиент к MCP-серверу памяти
      memory_service.py эмбеддер и экстрактор для памяти
      triplex/          ядро памяти (см. docs/MEMORY.md)
      unified_mcp/      единый MCP-сервер (см. docs/MCP.md)
      auth_*.py         вход по email и сессии
    web/                интерфейс чата и страницы входа
    tests/              тесты
    eval/               измерение качества поиска (G2, G3)
    docs/               документация

## Запуск (Windows)

1. Создайте окружение и поставьте зависимости:

       python -m venv .venv
       .venv\Scripts\pip install -r requirements.txt

2. Скопируйте `.env.example` в `.env` и заполните:
   - бэкенд модели (`LLM_BACKEND`, `MODELS_DIR` или адрес LM Studio);
   - `PULSENET_MCP_TOKEN` — сгенерируйте: `python -c "import secrets; print(secrets.token_urlsafe(32))"`;
   - `TRIPLEX_USER_NAME` — ваше имя (в памяти ваши реплики записываются под ним).
   Для локальной работы без входа укажите `AUTH_DISABLED=1`.

3. Запустите `run_server.bat`. Скрипт поднимает nginx и SearXNG (если работает Docker),
   локальный EasyDiffusion, MCP-сервер (порт 8010) и веб-сервер (порт 8000).
   Остановка всего: `run_server.bat stop`.

4. Откройте http://127.0.0.1:8000.

## Документация

- [docs/MEMORY.md](docs/MEMORY.md) — как устроена память: слои, время, инварианты, поиск, эмбеддер, «Сон».
- [docs/MCP.md](docs/MCP.md) — MCP-сервер: подключение клиентов, токены, инструменты, ошибки.
- [AGENTS.md](AGENTS.md) — правила работы агента с памятью (для opencode и других клиентов).
- [docs/AGENT_WORKSPACE.md](docs/AGENT_WORKSPACE.md) — навыки и заметки агента чата.
- [docs/TTS.md](docs/TTS.md) — озвучка: движки Piper и Silero, настройка, голоса.
- [eval/README.md](eval/README.md) — проверки качества поиска.
- [ROADMAP.md](ROADMAP.md) — открытые задачи.

## Лицензия

TBD — MIT или Apache 2.0. До выбора лицензии не принимайте код из внешних источников.
