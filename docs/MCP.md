# MCP-сервер pulsenet-unified

Единый MCP-сервер проекта: память, идентичность, общее ядро, поиск в интернете, генерация
изображений, чтение кода. Пакет `pulsenet/unified_mcp/`.

    python -m pulsenet.unified_mcp serve      # по умолчанию 127.0.0.1:8010, путь /mcp

Команду запускайте из **корня репозитория**: `python -m pulsenet.unified_mcp` и загрузка `.env` рассчитывают на текущую папку.

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
| `MCP_ALLOWED_ORIGINS` | принимаемые значения заголовка Origin, если клиент его шлёт |
| `MCP_MEMORY_SCOPE` | `shared` — одна память на всех агентов, `per_user` — отдельная на каждого (`shared`) |
| `MCP_TOKEN_SECRET` | секрет шифрования токенов; без него сервер создаёт `mcp_token.key` рядом с данными |
| `CODE_ACCESS_USERS` | кто читает код; пусто — никому |
| `CODE_ROOTS` | корни для чтения кода в виде `имя=путь`, несколько через `;` |
| `IDENTITY_MIN_DAYS` | сколько дней должна держаться формулировка личности (`2`) |
| `CORE_MIN_INTERVAL_HOURS` | минимальный интервал между изменениями общего ядра (`24`) |
| `SEARCH_BACKENDS_ORDER` | порядок поиска, побеждает первый с результатами (`searxng,ddg`) |
| `SEARXNG_URL` | адрес SearXNG, например `http://127.0.0.1:8080` |
| `SEARCH_TIME_BUDGET` | секунд на один поиск (`15`) |
| `WEB_CACHE_TTL` | срок кэша `web_fetch` в секундах (`600`) |
| `IMAGE_BACKENDS_ORDER` | порядок генерации изображений (`easydiffusion,pollinations`) |
| `IMAGE_OUTPUT_DIR` | папка картинок; по умолчанию **не** привязана к `PULSENET_DATA_DIR` |
| `IMAGE_INLINE_MAX_BYTES` | порог встроенной картинки в ответе (`1500000`) |
| `EASYDIFFUSION_*` | параметры локального рендера (адрес, число шагов и др.) |
| `BLOCKCOIN_PATH` | необязательно: путь к проекту, из которого подключается глубокий поиск |

Для чата в `.env` задайте `PULSENET_MEMORY=mcp`, `PULSENET_MCP_URL` (по умолчанию
`http://127.0.0.1:8010/mcp`) и `PULSENET_CHAT_MCP_TOKEN` — токен пользователя `chat`.

## Токены

- **Владелец** — `PULSENET_MCP_TOKEN` из `.env`. Чтобы сменить, задайте новое значение и перезапустите
  сервер. Команда `rotate owner` не работает: токен владельца в базу не пишется.
- **Остальные пользователи** — токены в базе `tokens.sqlite`. Хеш хранится всегда; при заданном
  `MCP_TOKEN_SECRET` (иначе сгенерированный `mcp_token.key`) токен хранится зашифрованным и может
  быть показан через `active()` / Настройки.

Команды (запускаются из **корня репозитория** с тем же `PULSENET_DATA_DIR`, что у сервера):

    python -m pulsenet.unified_mcp issue-token <имя>   # выпустить; значение показывается один раз
    python -m pulsenet.unified_mcp rotate <имя>        # заменить; старый токен перестаёт работать сразу
    python -m pulsenet.unified_mcp revoke <имя>        # отозвать без замены
    python -m pulsenet.unified_mcp list-tokens         # кто, когда, активен ли; значения не видны

Без `MCP_TOKEN_SECRET` потерянный токен восстановить нельзя — выпустите новый через `rotate`.
С заданным секретом текущий токен можно получить из Настроек.

## Подключение клиентов

**OpenCode V2.** Конфиг может быть пользовательским (`~/.config/opencode/opencode.json(c)`) или
project-local (`opencode.json(c)` / `.opencode/opencode.json(c)`). В этом репозитории файла конфига
нет: это настройка клиента, а не часть проекта.

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "pulsenet-unified": {
        "type": "remote",
        "url": "http://127.0.0.1:8010/mcp",
        "oauth": false,
        "headers": {
          "Authorization": "Bearer {env:PULSENET_MCP_TOKEN}"
        }
      }
    }
  }
}
```

`{env:NAME}` требует, чтобы переменная была в окружении OpenCode; вместо неё можно использовать
`{file:path}` с файлом, содержащим только токен. Поле `enabled` в V2 не используется — для отключения
есть `disabled: true`. Не подключайте один и тот же сервер дважды под разными именами: это удваивает
каталог инструментов.

**Клиенты только с stdio** (например, Claude Desktop) подключаются через мост `mcp-remote`
(нужен Node.js). В настройках клиента укажите команду `npx mcp-remote http://127.0.0.1:8010/mcp`
и заголовок `Authorization: Bearer <токен>`.

**Через nginx.** `nginx.conf` проксирует `/mcp` и `/mcp/` на сервер 8010 с `proxy_buffering off`.
Заголовок `Authorization` не сбрасывается. Домен добавьте в `MCP_ALLOWED_HOSTS`, при необходимости
Origin — в `MCP_ALLOWED_ORIGINS`. Раздача `/images/` проксируется отдельно.

**Проверка.** Вызов `tools/list` с токеном должен вернуть 33 инструмента:

    curl -s -X POST http://127.0.0.1:8010/mcp \
      -H "Authorization: Bearer <токен>" -H "Content-Type: application/json" \
      -H "Accept: application/json, text/event-stream" \
      -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'

## Инструменты (33)

**Память (`triplex_*`)**

| Инструмент | Что делает |
|---|---|
| `triplex_ingest` | сохранить текст как документ и извлечь факты; возвращает `doc_id` |
| `triplex_search` | гибридный поиск; возвращает факты, фрагменты и готовый `context` |
| `triplex_think` | пустить импульс по графу; возвращает `path`, `paths`, `alternatives`, `open_questions`, `thought_id`; есть `parent_thought_id` |
| `triplex_feedback` | подкрепить/ослабить путь: `fact_ids` из `thought.path`, `valence`, `thought_id`; или `verdicts=[{fact_id, valence, reason}]` + `outcome` |
| `triplex_thoughts` | история мыслей: `limit`, фильтр `goal`; мысли переживают перезапуск |
| `triplex_reflect` | разбор мысли по `thought_id`: все пути, альтернативы, открытые вопросы, обратная связь |
| `triplex_query` | действующие факты на момент `as_of` |
| `triplex_facts` | список действующих фактов с полем `agent` (кто записал) |
| `triplex_graph` | узлы и связи памяти (карта); при живом следе мысли — `path`, `paths`, `alternatives`, `subgoals`, `open_fact_ids`, `conflicts` |
| `triplex_assert` | записать факт; нужны `doc_id` и `span=[start,end]` из `triplex_ingest` |
| `triplex_supersede` | заменить факт новым; обязательны `valid_from`, `doc_id`, `span` |
| `triplex_retract` | отозвать (запись остаётся) |
| `triplex_confirm` | подтвердить или опровергнуть уже записанный факт (`fact_id`, `yes`) |
| `triplex_link` | привязать написание к существующей сущности |
| `triplex_define_relation` | онтология: синонимы, функциональность, срок затухания |
| `triplex_stats` | счётчики |
| `triplex_check` | проверка инвариантов журнала |
| `triplex_consolidate` | «Сон», сухой прогон: без обучения |

Подробности фазы 4 — в `docs/PHASE4.md`.

**Идентичность и общее ядро**

| Инструмент | Что делает |
|---|---|
| `identity_propose` | предложить формулировку о себе; нужна цитата из памяти |
| `identity_commit` | принять предложение (после `IDENTITY_MIN_DAYS` дней) |
| `identity_chain` | история формулировок |
| `core_get` | общее ядро (читают все) |
| `core_propose`, `core_commit` | изменить ядро — только администраторы (`MCP_ADMIN_USERS`) |

**Интернет:** `web_search` (`query` **или** `queries`, до 4 запросов, `max_results` 1–10), `web_fetch`
(только публичные http(s) без редиректов; защита от SSRF). Результаты кэшируются на `WEB_CACHE_TTL` секунд.

**Изображения:** `image_generate` — параметры: `prompt`, `seed`, `steps` (1–60), `width`/`height`
(64–1536, кратно 8), `cfg_scale` (1–30), `sampler`, `negative_prompt`, `backends`, `return_image`.
Бэкенды перебираются по порядку, побеждает первый успешный. PNG сохраняется в
`IMAGE_OUTPUT_DIR/<пользователь>/`, рядом лежит `.json` с параметрами. Ответ содержит картинку и JSON
с метаданными (`status`, `backend`, `file_path`, `url`, `seed`, `attempts`). Если
`return_image: false` или файл больше `IMAGE_INLINE_MAX_BYTES` — только JSON.

**Код:** `code_list(root, subdir)`, `code_read(root, path, start, end)`, `code_search(root, pattern)` —
только для `CODE_ACCESS_USERS` и только внутри `CODE_ROOTS`; `.env` и секреты запрещены, чтение до 300 строк.

**Служебные:** `activity_recent(limit)` (журнал действий), `pulsenet_guide` (описание сервера и порядок работы;
вызывайте первым, если не уверены), `whoami`.

## Ошибки

- `401` — токен не передан, неверный или отозван.
- `403` — заголовок `Origin` не разрешён (`Invalid Origin header`).
- `421` — заголовок `Host` не разрешён: защита от DNS-rebinding в SDK MCP (`Misdirected Request`).
- `503` — `PULSENET_MCP_TOKEN` не задан, `/mcp` отключён.
- Ошибки логики (например, неизвестный `fact_id`) возвращаются как результат инструмента с `status: "error"`,
  а не как HTTP-ошибка.
- Если MCP недоступен, чат продолжает работу без памяти; индексация сообщений повторяет попытки.

## Ограничения

- SearXNG должен отдавать JSON: в `searxng/settings.yml` формат `json` уже включён.
  DuckDuckGo — запасной вариант, его разметка может меняться.
- Защита от SSRF проверяет адрес в момент запроса; подмену DNS между проверкой и загрузкой она не закрывает полностью.
- Эмбеддер по умолчанию — офлайн-хеш. Для смыслового поиска подключите модель (см. docs/MEMORY.md).
- Выдача токенов — только через CLI на сервере; веб-страницы для этого нет.
- `MCP_MEMORY_SCOPE=shared` — одна память для **всех токенов/пользователей сервера**. Изоляция только при `per_user`.
- `/images/<user>/<file>.png` раздаётся без авторизации (middleware защищает только `/mcp`). Не публикуйте приватные картинки.
- `IMAGE_OUTPUT_DIR` по умолчанию не зависит от `PULSENET_DATA_DIR`; при смене каталога данных задайте его явно.
