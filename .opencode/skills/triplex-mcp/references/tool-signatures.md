# pulsenet-unified: 31 инструмент

Источник — `pulsenet/unified_mcp/server.py` и живой `tools/list`. Если подпись здесь расходится с `tools/list`, доверяйте `tools/list`.

## Память (`triplex_*`)

| Инструмент | Аргументы |
|---|---|
| `triplex_ingest` | `text` (обязательно); `session_id=""`, `role="user"`, `model=""`, `recorded_at=""` |
| `triplex_search` | `query` (обязательно); `k=8` (1–50), `as_of=null`, `include_archived=false`, `goal=""`, `activate=true`, `remember_thought=true` |
| `triplex_think` | `query` (обязательно); `goal=""`, `k=8`, `steps=3` (1–8), `as_of=null` |
| `triplex_feedback` | `fact_ids` (список из `thought.path`), `valence` (-1…1), `thought_id` из результата `triplex_think` |
| `triplex_query` | `s`, `r`, `o`, `as_of` — все необязательны |
| `triplex_assert` | `s`, `r`, `o`, `doc_id`, `span=[start,end]`; `confidence=0.8`, `functional=false` |
| `triplex_supersede` | `old_fact_id`, `s`, `r`, `o`, `valid_from` (ISO 8601), `doc_id`, `span`; `confidence=0.8`, `functional=false` |
| `triplex_facts` | `limit=50` |
| `triplex_link` | `entity_id`, `alias` |
| `triplex_retract` | `fact_id`; `at` по желанию |
| `triplex_confirm` | `fact_id`, `yes` (+0.15 или ×0.5 к confidence) |
| `triplex_define_relation` | `canonical`, `aliases`, `functional`, `tau_days` |
| `triplex_stats` | без аргументов |
| `triplex_check` | без аргументов |
| `triplex_consolidate` | сухой прогон отбора фактов; обучение отдельно: `python -m pulsenet train-sleep` |
| `triplex_graph` | `max_edges=200` |

`span` — всегда два целых числа: позиция начала и конца подстроки в тексте, сохранённом `triplex_ingest`. Объект `{source: ...}` передавать нельзя.

## Идентичность и ядро

| Инструмент | Аргументы / правила |
|---|---|
| `identity_propose` | `content` (30–600), `kind` (одно из 5 значений из `store.py`), `evidence_quote` — цитата из памяти |
| `identity_commit` | `candidate_id`; карантин `IDENTITY_MIN_DAYS` по умолчанию 2 дня |
| `identity_chain` | без аргументов |
| `core_get` | без аргументов, читают все |
| `core_propose` | только администраторы из `MCP_ADMIN_USERS` |
| `core_commit` | только администраторы, не чаще `CORE_MIN_INTERVAL_HOURS` (24 ч) |

## Интернет

| Инструмент | Аргументы |
|---|---|
| `web_search` | `query` **или** `queries` (до 4), `max_results=5` (1–10) |
| `web_fetch` | `url`; только публичные http(s), без редиректов; кэш `WEB_CACHE_TTL` (600 с) |

## Изображения

| Инструмент | Аргументы |
|---|---|
| `image_generate` | `prompt`; `steps` 1–60, `width`/`height` 64–1536 (кратно 8), `cfg_scale` 1–30, `seed` (-1 = случайно), `sampler`, `negative_prompt`, `backends`, `return_image=true` |

Бэкенды перебираются по порядку, побеждает первый успешный. PNG и `.json` с параметрами — в `IMAGE_OUTPUT_DIR/<user>/`. Встроенная картинка не приходит, если файл больше `IMAGE_INLINE_MAX_BYTES` (1.5 МБ) или передан `return_image: false`.

## Код

| Инструмент | Аргументы |
|---|---|
| `code_list` | `root` (обязателен, из `CODE_ROOTS`), `subdir=""` |
| `code_read` | `root`, `path`; `start=1`, `end=null` (до 300 строк за вызов) |
| `code_search` | `root`, `pattern`; поиск без учёта регистра |

Доступ только для пользователей из `CODE_ACCESS_USERS`. Песочница: `realpath` внутри `CODE_ROOTS`, `.env` и секреты запрещены. Чат скрывает `code_*`.

## Служебные

| Инструмент | Аргументы |
|---|---|
| `activity_recent` | `limit=30` |
| `pulsenet_guide` | без аргументов |
| `whoami` | без аргументов |

`ctx` в аргументах нет: OpenCode передаёт ID сессии в `_meta["ai.opencode/sessionID"]`.
