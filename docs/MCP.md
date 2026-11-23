# MCP-сервер pulsenet-unified

Единый MCP-сервер проекта: память, идентичность, общее ядро, поиск в интернете, генерация
изображений, чтение кода. Пакет `pulsenet/unified_mcp/`.

    python -m pulsenet.unified_mcp serve      # по умолчанию 127.0.0.1:8010, путь /mcp

Транспорт: Streamable HTTP, без состояния, ответы в JSON. Протокол — официальный SDK `mcp` (версия 1.x).
Это единственный MCP-сервер проекта: чат приложения и внешние агенты работают с одним и тем же хранилищем.

## Настройка

Переменные из `.env` (значения по умолчанию в скобках):

| Переменная | Назначение |
|---|---|
| `PULSENET_UNIFIED_HOST`, `PULSENET_UNIFIED_PORT` | адрес сервера (`127.0.0.1`, `8010`) |
| `PULSENET_DATA_DIR` | папка данных; веб-сервер и MCP должны использовать одну и ту же (`data_unified`) |
| `PULSENET_MCP_TOKEN` | токен владельца |
| `MCP_OWNER_ID` | имя владельца (`owner`) |
| `MCP_ADMIN_USERS` | кто меняет общее ядро (через запятую) |
| `MCP_ALLOWED_HOSTS` | принимаемые значения заголовка Host; для домена через nginx добавьте его сюда |
| `MCP_MEMORY_SCOPE` | `shared` — одна память на всех агентов, `per_user` — отдельная на каждого (`shared`) |
| `CODE_ACCESS_USERS` | кто читает код; пусто — никому |
| `CODE_ROOTS` | корни для чтения кода в виде `имя=путь`, несколько через `;` |
| `IDENTITY_MIN_DAYS` | сколько дней должна держаться формулировка личности (`2`) |
| `CORE_MIN_INTERVAL_HOURS` | минимальный интервал между изменениями общего ядра (`24`) |
| `SEARCH_BACKENDS_ORDER` | порядок поиска, побеждает первый с результатами (`searxng,ddg`) |
| `SEARXNG_URL` | адрес SearXNG, например `http://127.0.0.1:8080` |
| `SEARCH_TIME_BUDGET` | секунд на один поиск (`15`) |
| `IMAGE_BACKENDS_ORDER` | порядок генерации изображений (`easydiffusion,pollinations`) |
| `BLOCKCOIN_PATH` | необязательно: путь к проекту, из которого подключается глубокий поиск |

Для чата в `.env` задайте `PULSENET_MEMORY=mcp`, `PULSENET_MCP_URL` (по умолчанию
`http://127.0.0.1:8010/mcp`) и `PULSENET_CHAT_MCP_TOKEN` — токен пользователя `chat`.

## Токены

- **Владелец** — `PULSENET_MCP_TOKEN` из `.env`. Чтобы сменить, задайте новое значение и перезапустите
  сервер. Команда `rotate owner` не работает: токен владельца в базе не хранится.
- **Остальные пользователи** — токены в базе `tokens.sqlite` (в базе только хеш).

Команды (запускаются из папки `pulsenet` с тем же `PULSENET_DATA_DIR`, что у сервера):

    python -m pulsenet.unified_mcp issue-token <имя>   # выпустить; значение показывается один раз
    python -m pulsenet.unified_mcp rotate <имя>        # заменить; старый токен перестаёт работать сразу
    python -m pulsenet.unified_mcp revoke <имя>        # отозвать без замены
    python -m pulsenet.unified_mcp list-tokens         # кто, когда, активен ли; значения не видны

Потерянный токен восстановить нельзя — выпустите новый через `rotate`.

## Подключение клиентов

**opencode.** Файл `opencode.json` в корне проекта. Токен берётся из переменной окружения:

    {
      "$schema": "https://opencode.ai/config.json",
      "mcp": {
        "pulsenet": {
          "type": "remote",
          "url": "http://127.0.0.1:8010/mcp",
          "headers": { "Authorization": "Bearer {env:PULSENET_MCP_TOKEN}" },
          "enabled": true
        }
      }
    }

**Клиенты только с stdio** (например, Claude Desktop) подключаются через мост `mcp-remote`
(нужен Node.js). В настройках клиента укажите команду `npx mcp-remote http://127.0.0.1:8010/mcp`
и заголовок `Authorization: Bearer <токен>`.

**Через nginx.** `nginx.conf` проксирует `/mcp` на сервер 8010 с `proxy_buffering off`. Заголовок
`Authorization` не сбрасывается. Домен добавьте в `MCP_ALLOWED_HOSTS`.

**Проверка.** Вызов `tools/list` с токеном должен вернуть список инструментов ниже:

    curl -s -X POST http://127.0.0.1:8010/mcp \
      -H "Authorization: Bearer <токен>" -H "Content-Type: application/json" \
      -H "Accept: application/json, text/event-stream" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

## Инструменты (29)

**Память (`triplex_*`)**

| Инструмент | Что делает |
|---|---|
| `triplex_ingest` | сохранить текст как документ и извлечь факты; возвращает `doc_id` |
| `triplex_search` | гибридный поиск; возвращает факты, фрагменты и готовый `context` |
| `triplex_query` | действующие факты на момент `as_of` |
| `triplex_facts` | список действующих фактов с полем `agent` (кто записал) |
| `triplex_graph` | узлы и связи памяти (карта) |
| `triplex_assert` | записать факт; нужны `doc_id` и `span` из `triplex_ingest` |
| `triplex_supersede` | заменить факт новым |
| `triplex_retract` | отозвать (запись остаётся) |
| `triplex_confirm` | подтвердить или опровергнуть факт |
| `triplex_link` | привязать написание к существующей сущности |
| `triplex_define_relation` | онтология: синонимы, функциональность, срок затухания |
| `triplex_stats` | счётчики |
| `triplex_check` | проверка инвариантов журнала |
| `triplex_consolidate` | «Сон», сухой прогон: без обучения |

**Идентичность и общее ядро**

| Инструмент | Что делает |
|---|---|
| `identity_propose` | предложить формулировку о себе; нужна цитата из памяти |
| `identity_commit` | принять предложение (после `IDENTITY_MIN_DAYS` дней) |
| `identity_chain` | история формулировок |
| `core_get` | общее ядро (читают все) |
| `core_propose`, `core_commit` | изменить ядро — только администраторы (`MCP_ADMIN_USERS`) |

**Интернет:** `web_search` (до 4 запросов параллельно), `web_fetch` (только публичные http(s) без
редиректов; защита от SSRF). Результаты кэшируются на `WEB_CACHE_TTL` секунд (по умолчанию 600).

**Изображения:** `image_generate` — параметры: `prompt`, `seed`, `steps`, `width`, `height`,
`negative_prompt`, `backends`, `return_image`. Бэкенды перебираются по порядку, побеждает первый
успешный. PNG сохраняется в `IMAGE_OUTPUT_DIR/<пользователь>/`, рядом лежит `.json` с параметрами.
Ответ содержит картинку и JSON с метаданными (`status`, `backend`, `file_path`, `url`, `seed`, `attempts`).
Если `return_image: false` — только JSON.

**Код:** `code_list`, `code_read`, `code_search` — только для `CODE_ACCESS_USERS` и только внутри `CODE_ROOTS`.

**Служебные:** `activity_recent` (журнал действий), `pulsenet_guide` (описание сервера и порядок работы;
вызывайте первым, если не уверены), `whoami`.

## Ошибки

- `401` — токен не передан, неверный или отозван.
- Заголовок `Host`, которого нет в `MCP_ALLOWED_HOSTS`, отклоняется.
- Ошибки логики (например, неизвестный `fact_id`) возвращаются как результат инструмента с `status: "error"`,
  а не как HTTP-ошибка.
- Если MCP недоступен, чат продолжает работу без памяти; индексация сообщений повторяет попытки.

## Ограничения

- SearXNG должен отдавать JSON: в `searxng/settings.yml` формат `json` уже включён.
  DuckDuckGo — запасной вариант, его разметка может меняться.
- Защита от SSRF проверяет адрес в момент запроса; подмену DNS между проверкой и загрузкой она не закрывает полностью.
- Эмбеддер по умолчанию — офлайн-хеш. Для смыслового поиска подключите модель (см. docs/MEMORY.md).
- Выдача токенов — только через CLI на сервере; веб-страницы для этого нет.
