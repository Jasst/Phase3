# -*- coding: utf-8 -*-
"""Протокол вызова инструментов моделью — текстовый, не зависит от бэкенда.

Модель пишет блок <tool_call>{"name": "...", "arguments": {...}}</tool_call> и останавливается.
Приложение исполняет вызов и отдаёт результат в теге <tool_result name="...">...</tool_result>.
Так работает любой OpenAI-совместимый сервер (llama-server, LM Studio), даже без поддержки
нативного function calling. Инструменты описаны в tools.py.

Команды в чате (`/search`, `/fetch`, `/memory`, `/remember`) выполняются приложением напрямую:
модели передаётся уже готовый результат, а не просьба вызвать инструмент.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

OPEN = "<tool_call>"
CLOSE = "</tool_call>"

# Gemma 4 (и некоторые другие локальные модели) выдают вызовы служебными токенами
# с несимметричными скобками: <|tool_call>call:web_search(queries=["a"])<tool_call|>.
# LM Studio отдаёт это обычным текстом в delta.content, поэтому формат надо
# разбирать здесь, а не ждать native_tool_calls от сервера.
NATIVE_OPEN = "<|tool_call>"
NATIVE_CLOSE = ("<tool_call|>", "<|/tool_call|>", "<|tool_call_end|>")
_NATIVE_OPEN_RE = re.compile(r"^<\s*\|\s*tool_call\s*(?:\|>|>)", re.I)
_NATIVE_CLOSE_RE = re.compile(
    r"^(?:<\s*tool_call\s*\|>|<\s*\|\s*/?\s*tool_call\s*\|>|<\s*\|\s*tool_call_end\s*\|>)", re.I)


def _env_int(env: dict, key: str, default: int, lo: int = 1, hi: int = 100) -> int:
    try:
        return max(lo, min(hi, int(str(env.get(key, "")).strip() or default)))
    except ValueError:
        return default


def _env_int0(env: dict, key: str, default: int, hi: int = 100000) -> int:
    """Как _env_int, но 0 — осознанный выбор «без лимита», а не опечатка."""
    try:
        value = int(str(env.get(key, "")).strip() or default)
    except ValueError:
        return default
    return max(0, min(hi, value))


def agent_limits(env: Optional[dict] = None) -> dict:
    """Лимиты агентного цикла. 0 = без лимита; env=None — текущее окружение."""
    env = os.environ if env is None else env
    rounds = _env_int0(env, "AGENT_MAX_ROUNDS", 6, 7)
    return {
        "max_rounds": rounds,
        # Оставлен для совместимости со старыми .env; в цикле не используется.
        "hard_cap": _env_int(env, "AGENT_HARD_CAP", max(rounds + 2, 8), 1, 12),
        "max_calls": _env_int0(env, "AGENT_MAX_CALLS", 12, 40),
        "reflection_budget": _env_int(env, "AGENT_REFLECTION_BUDGET", 2, 0, 10),
        # Сколько раз просить модель договорить после finish_reason=length. 0 — без лимита.
        "max_continuations": _env_int0(env, "AGENT_MAX_CONTINUATIONS", 2, 1000),
        "result_chars": _env_int(env, "TOOL_RESULT_CONTEXT_CHARS", 1200, 200, 20000),
        # Отдельный лимит генерации на шаг агента: 1024 не хватало на JSON <tool_call>
        "agent_max_tokens": _env_int(env, "AGENT_MAX_TOKENS", 2048, 512, 8192),
        # Потолки системного промпта и всего запроса, чтобы локальная модель не обрезалась
        "system_max": _env_int(env, "AGENT_SYSTEM_MAX_CHARS", 6000, 1000, 30000),
        "context_max": _env_int(env, "AGENT_CONTEXT_MAX_CHARS", 20000, 4000, 60000),
        # 0 — отправлять все доступные схемы, а не только первые N.
        "native_max": _env_int0(env, "TOOL_NATIVE_MAX", 10, 30),
        "image_timeout": _env_int(env, "MCP_TOOL_TIMEOUT_IMAGE", 300, 30, 900),
        # параллельные вызовы в одном шаге: сколько принимаем и сколько выполняем одновременно
        "calls_per_step": _env_int0(env, "AGENT_CALLS_PER_STEP", 4, 8) or 1000000,
        "parallel": _env_int(env, "AGENT_PARALLEL", 4, 1, 8),
        # субагент: 0 раундов = работает, пока не вернёт текст
        "sub_rounds": _env_int0(env, "SUBAGENT_MAX_ROUNDS", 3, 6),
        "llm_parallel": _env_int(env, "AGENT_LLM_PARALLEL", 1, 1, 4),
        # пауза после сбоя сервера памяти: ход не ждёт его каждый раз (сек)
        "mcp_cooldown": _env_int(env, "MCP_COOLDOWN_SECONDS", 45, 0, 600),
    }


LIMITS = agent_limits()
MAX_STEPS = LIMITS["max_rounds"]   # сколько раз модель может вызвать инструменты за один ответ
MAX_CALLS_PER_STEP = LIMITS["calls_per_step"]   # сколько вызовов за один шаг принимаем (как в AGENT_CALLS_PER_STEP)
MAX_RESULT_CHARS = 6000            # потолок текста результата, который уходит в контекст модели

# Команды: ключ — имя без слэша, usage — подсказка, local — выполняется только в интерфейсе.
COMMANDS: Dict[str, dict] = {
    "search": {"usage": "/search запрос — поиск в интернете и ответ по источникам",
               "hint": "поиск в интернете", "local": False},
    "fetch": {"usage": "/fetch адрес — прочитать страницу и ответить по ней",
              "hint": "прочитать страницу по адресу", "local": False},
    "memory": {"usage": "/memory запрос — найти в памяти факты и фрагменты",
               "hint": "поиск по памяти", "local": False},
    "remember": {"usage": "/remember текст — сохранить в память",
                 "hint": "сохранить в память", "local": False},
    "chatmem": {"usage": "/chatmem — загрузить текущий чат в память как документ и извлечь факты",
                "hint": "чат → память (с дедупликацией)", "local": True},
    "graph": {"usage": "/graph — открыть карту нейронов памяти",
              "hint": "карта нейронов", "local": True},
    "new": {"usage": "/new — новый чат", "hint": "новый чат", "local": True},
}

_COMMAND_RE = re.compile(r"^/([a-z]+)(?:\s+(.*))?$", re.S | re.I)


def parse_command(message: str) -> Optional[Tuple[str, str]]:
    """('search', 'текст запроса') для «/search текст запроса»; None — обычное сообщение.
    Неизвестные команды считаются обычным текстом (например, пути вида /usr/bin)."""
    m = _COMMAND_RE.match((message or "").strip())
    if not m:
        return None
    name = m.group(1).lower()
    if name not in COMMANDS or COMMANDS[name]["local"]:
        return None
    return name, (m.group(2) or "").strip()


def _tag_name(tag: str) -> str:
    m = re.match(r"<\s*/?\s*([A-Za-z_][\w-]*)", tag or "")
    return m.group(1).lower() if m else ""


# Основной аргумент инструмента для тегового вызова: <web_search>запрос</web_search>.
# Для MCP-инструментов с другим именем аргумента app передаёт свою карту primary.
_XML_PRIMARY = {
    "memory_search": "query", "memory_remember": "text", "web_search": "queries",
    "web_fetch": "url", "skill_read": "name", "skill_write": "name", "skill_patch": "name",
    "note_write": "title", "note_append": "title", "note_read": "title", "note_list": "",
    "image_generate": "prompt", "subagent": "task", "todo_write": "items",
}


def _loose_json(text: str):
    """JSON из тела тега, даже если вокруг него ```json или лишний текст. None — не JSON."""
    body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (text or "").strip(), flags=re.I)
    if not body:
        return None
    try:
        return json.loads(body)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(body):
        if ch in "[{":
            try:
                value, _ = decoder.raw_decode(body, i)
            except ValueError:
                continue
            return value
    return None


def _xml_call_body(name: str, raw: str, primary: Optional[Dict[str, str]] = None) -> str:
    """Тело <web_search>...</web_search> → канонический JSON вида {"name", "arguments"}.

    Модели пишут по-разному: массив запросов, объект аргументов, просто строку.
    Приводим значение к основному аргументу инструмента, чтобы вызов не пропал."""
    text = str(raw or "").strip()
    value = _loose_json(text) if text else None
    args_map = dict(_XML_PRIMARY)
    args_map.update(primary or {})
    arg = args_map.get(name) or "query"
    if isinstance(value, dict):
        args = value
    elif value is None:
        args = {arg: text} if text and arg else {}
    elif arg == "queries":
        items = value if isinstance(value, list) else [value]
        queries = [str(v).strip() for v in items if str(v).strip()]
        args = {"queries": queries[:3]} if queries else {}
    elif isinstance(value, list) and arg == "query":
        queries = [str(v).strip() for v in value if str(v).strip()]
        args = {"query": " ".join(queries)} if queries else {}
    elif isinstance(value, list) and name == "todo_write":
        args = {"items": [{"text": str(v), "status": "pending"} if isinstance(v, str) else v
                          for v in value]}
    elif isinstance(value, list):
        args = {arg: value}
    else:
        args = {arg: value if isinstance(value, str) else str(value)}
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


# ---------- нативный протокол Gemma 4: <|tool_call>call:name(k=v)<tool_call|> ----------

def _native_open_at(text: str, pos: int = 0) -> int:
    """Длина открывающего служебного токена в text[pos:] (0 — его нет)."""
    m = _NATIVE_OPEN_RE.match(text, pos)
    return m.end() - pos if m else 0


def _native_close_at(text: str, pos: int = 0) -> int:
    """Длина закрывающего служебного токена в text[pos:] (0 — его нет)."""
    m = _NATIVE_CLOSE_RE.match(text, pos)
    return m.end() - pos if m else 0


def _possible_native(tail: str) -> bool:
    """Похож ли хвост на начало служебного токена (токены приходят по частям)."""
    norm = re.sub(r"\s+", "", tail).lower()
    if not norm.startswith("<"):
        return False
    markers = ("<|tool_call>", "<|tool_call|>", "<tool_call|>", "<|/tool_call|>", "<|tool_call_end|>")
    return any(m.startswith(norm) for m in markers)


def _native_may_continue(tail: str) -> bool:
    """Ещё неполный вызов: имя и аргументы могут прийти несколькими кусками."""
    norm = re.sub(r"\s+", "", tail).lower()
    if not norm or "call:".startswith(norm):
        return True
    # Пока не закрылись скобки, держим весь хвост в буфере — иначе токены call:...
    # просочатся пользователю. Конец потока обработает finish().
    return bool(re.match(r"^call:[A-Za-z_][\w.-]*[\(\{]?$", norm)
                or re.match(r"^call:[A-Za-z_][\w.-]*[\(\{]", norm))


def _native_call_end(text: str, start: int) -> int:
    """Индекс после балансированного call:name(...) / call:name{...}. -1 — вызов не полон."""
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    m = re.match(r"call\s*:\s*", text[i:], re.I)
    if not m:
        return -1
    i += m.end()
    m = re.match(r"[A-Za-z_][\w.-]*", text[i:])
    if not m:
        return -1
    i += m.end()
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] not in "([{":
        return -1
    stack = [text[i]]
    i += 1
    quote = ""
    while i < len(text):
        ch = text[i]
        if text.startswith('<|"|>', i) or text.startswith("<|'|>", i):
            # Gemma экранирует кавычки служебным токеном <|"|> / <|'|> (5 символов).
            qch = text[i + 2]
            if not quote:
                quote = qch
            # внутри строки это экранированная кавычка — строку не закрываем
            i += 5
            continue
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack:
                return -1
            stack.pop()
            if not stack:
                return i + 1
        i += 1
    return -1


def _split_top(text: str) -> List[str]:
    """Разделить аргументы по запятым вне строк и вложенных скобок."""
    parts: List[str] = []
    buf: List[str] = []
    stack: List[str] = []
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if text.startswith('<|"|>', i) or text.startswith("<|'|>", i):
            buf.append('"')
            i += 5
            continue
        if quote:
            buf.append(ch)
            if ch == "\\" and i + 1 < len(text):
                buf.append(text[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if stack:
                stack.pop()
        elif ch == "," and not stack:
            parts.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf or parts:
        parts.append("".join(buf).strip())
    return [p for p in parts if p]


def _top_separator(text: str) -> Optional[Tuple[int, str]]:
    """Позиция первого разделителя key=value / key: value вне строк и скобок."""
    stack: List[str] = []
    quote = ""
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if stack:
                stack.pop()
        elif ch in "=:" and not stack:
            return i, ch
        i += 1
    return None


def _native_value(raw: str):
    """Значение аргумента: JSON-литерал, Python-литерал или строка."""
    text = str(raw or "").strip()
    if not text:
        return ""
    if text[0] in ("'", '"') and len(text) > 1 and text[-1] == text[0]:
        return text[1:-1]
    try:
        return json.loads(text)
    except ValueError:
        pass
    plain = re.sub(r"\b(None|True|False)\b",
                   lambda m: {"None": "null", "True": "true", "False": "false"}[m.group(1)], text)
    try:
        return json.loads(plain)
    except ValueError:
        return text


def _native_args(name: str, inner: str, primary: Optional[Dict[str, str]] = None) -> dict:
    parts = _split_top(inner)
    if not parts:
        return {}
    if len(parts) == 1 and _top_separator(parts[0]) is None:
        # Один позиционный аргумент: кладём его в основной аргумент инструмента
        # (queries для поиска, prompt для генерации и т. д.).
        canon = _xml_call_body(name, parts[0], primary)
        try:
            return json.loads(canon).get("arguments") or {}
        except ValueError:
            return {}
    args: Dict[str, object] = {}
    for part in parts:
        sep = _top_separator(part)
        if sep is None:
            continue
        key = part[:sep[0]].strip().strip("'\"")
        if key:
            args[key] = _native_value(part[sep[0] + 1:])
    return args


def _native_call_body(raw: str, primary: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Тело нативного вызова Gemma → канонический JSON вызова (или None)."""
    text = str(raw or "").strip()
    text = re.sub(r"^(?:<\s*\|\s*tool_call\s*(?:\|>|>)|\s)+", "", text)
    text = re.sub(r"(?:<\s*tool_call\s*\|>|<\s*\|\s*/?\s*tool_call\s*\|?>|\s)+$", "", text)
    # Внутри JSON-строки служебный токен кавычки должен стать экранированной кавычкой.
    text = text.replace('<|"|>', '\\"').replace("<|'|>", "'")
    m = re.match(r"call\s*:\s*([A-Za-z_][\w.-]*)\s*([\(\{])", text, re.I | re.S)
    if not m:
        return None
    end = _native_call_end(text, 0)
    if end < 0 or end <= m.end():
        return None
    name, bracket = m.group(1), m.group(2)
    inner = text[m.end():end - 1]
    if bracket == "{":
        obj = _loose_json(inner)
        args = obj if isinstance(obj, dict) else _native_args(name, inner, primary)
    else:
        args = _native_args(name, inner, primary)
    return json.dumps({"name": name, "arguments": args}, ensure_ascii=False)


def strip_native_calls(text: str) -> str:
    """Убрать утечки служебных токенов нативных вызовов из итогового текста."""
    src = str(text or "")
    out: List[str] = []
    pos = 0
    while pos < len(src):
        m = re.search(r"<\s*\|\s*tool_call", src[pos:], re.I)
        if not m:
            out.append(src[pos:])
            break
        start = pos + m.start()
        out.append(src[pos:start])
        opened = _native_open_at(src, start)
        end = _native_call_end(src, start + opened) if opened else -1
        if end < 0:
            close = re.search(r"<\s*tool_call\s*\|>|<\s*\|\s*/?\s*tool_call\s*\|>", src[start:], re.I)
            end = start + (close.end() if close else len(src) - start)
        pos = end
    result = "".join(out)
    for marker in NATIVE_CLOSE:
        result = re.sub(re.escape(marker), "", result, flags=re.I)
    # Gemma 4 добавляет пустой канонический блок сразу после нативного вызова.
    result = re.sub(r"<\s*tool_call\s*>\s*</\s*tool_call\s*>", "", result, flags=re.I)
    return result


class ToolCallFilter:
    """Потоковый фильтр: отдаёт пользователю видимый текст и отделяет вызовы инструментов.

    Понимает канонический <tool_call>{"name": ...}</tool_call>, теговый вариант
    <web_search>...</web_search> и нативный формат Gemma 4
    <|tool_call>call:web_search(queries=["..."])<tool_call|> — LM Studio отдаёт его
    обычным текстом. feed() возвращает видимую часть очередного куска, finish() —
    остаток после конца потока. Тела вызовов лежат в calls_raw уже каноническим JSON,
    поэтому parse_calls разбирает все варианты одинаково. Незакрытый блок тоже попадает
    туда (на случай обрыва по лимиту токенов)."""

    def __init__(self, names: Iterable[str] = (), primary: Optional[Dict[str, str]] = None):
        self.buf = ""
        self.inside = False
        self.native = False          # внутри нативного вызова <|tool_call>call:...
        self.open_name = ""          # имя тега тегового вызова; "" — канонический <tool_call>
        self._open_tag = ""
        self.names = {str(n) for n in (names or ()) if n}
        self.primary = dict(primary or {})
        self.calls_raw: List[str] = []
        self.fake_results = 0        # сколько поддельных <tool_result> модель написала сама

    def feed(self, text: str) -> str:
        self.buf += text or ""
        out: List[str] = []
        while self.buf:
            if self.inside:
                if not self._consume_call():
                    break
            elif self.native:
                if not self._consume_native():
                    break
            elif not self._consume_text(out):
                break
        return "".join(out)

    def reset(self) -> None:
        """Очистить состояние перед повторной отправкой того же запроса модели.

        При обрыве потока в буфере может остаться начало тега или вызова; без сброса
        повторный ответ склеивается с этим мусором и выдаёт испорченный текст.
        """
        self.buf = ""
        self.inside = False
        self.native = False
        self.open_name = ""
        self._open_tag = ""
        self.calls_raw.clear()
        self.fake_results = 0

    def finish(self) -> str:
        rest, self.buf = self.buf, ""
        if self.inside:
            name, self.open_name = self.open_name, ""
            self.inside = False
            if name:
                self.calls_raw.append(_xml_call_body(name, rest, self.primary))
            elif rest.strip():
                self.calls_raw.append(rest)
            return ""
        if self.native:
            self.native = False
            # Поток оборвался посреди вызова: пробуем собрать его по открытым скобкам.
            raw = "<|tool_call>" + rest
            canon = _native_call_body(raw, self.primary)
            if canon:
                self.calls_raw.append(canon)
            return ""                 # служебные токены никогда не показываем пользователю
        return rest

    def _consume_text(self, out: List[str]) -> bool:
        """False — данных мало, ждём; True — нашли начало вызова."""
        while True:
            i = self.buf.find("<")
            if i < 0:
                out.append(self.buf)
                self.buf = ""
                return False
            if i:
                out.append(self.buf[:i])
                self.buf = self.buf[i:]
            close = _native_close_at(self.buf)
            if close:
                self.buf = self.buf[close:]
                continue
            opened = _native_open_at(self.buf)
            if opened:
                self.buf = self.buf[opened:]
                self.native = True
                return True
            j = self.buf.find(">")
            if j < 0:
                if (self._possible_tag(self.buf) or self._possible_tool_result(self.buf)
                        or _possible_native(self.buf)):
                    return False
                out.append(self.buf[0])          # не тег инструмента — обычный текст
                self.buf = self.buf[1:]
                continue
            tag = self.buf[:j + 1]
            if tag.startswith("</"):
                out.append(tag)
                self.buf = self.buf[j + 1:]
                continue
            name = _tag_name(tag)
            if name == "tool_result":
                consumed = self._consume_tool_result()
                if consumed is None:
                    return False
                if consumed:
                    continue
            if name == "tool_call" or name in self.names:
                self._open_tag = tag
                self.open_name = "" if name == "tool_call" else name
                self.inside = True
                self.buf = self.buf[j + 1:]
                return True
            out.append(tag)
            self.buf = self.buf[j + 1:]

    def _consume_native(self) -> bool:
        """Прочитать один вызов <|tool_call>call:name(...) между служебными токенами."""
        close = _native_close_at(self.buf)
        if close:
            self.buf = self.buf[close:]
            self.native = False
            return True
        if re.match(r"^<\s*tool_call\s*>", self.buf, re.I):
            # Модель добавила пустой канонический блок после нативного вызова.
            self.native = False
            return True
        end = _native_call_end(self.buf, 0)
        if end > 0:
            raw = self.buf[:end]
            self.buf = self.buf[end:]
            self.native = False
            canon = _native_call_body(raw, self.primary)
            self.calls_raw.append(canon or raw)
            return True
        if _native_may_continue(self.buf.lstrip()) or _possible_native(self.buf):
            return False
        # Вместо вызова пошло обычное объяснение — не теряем его.
        self.native = False
        return True

    def _consume_call(self) -> bool:
        close = "</" + (self.open_name or "tool_call") + ">"
        i = self.buf.lower().find(close)
        if i < 0:
            return False
        body, self.buf = self.buf[:i], self.buf[i + len(close):]
        self.inside = False
        name, self.open_name = self.open_name, ""
        tag, self._open_tag = self._open_tag, ""
        if name:
            self.calls_raw.append(_xml_call_body(name, body, self.primary))
        else:
            forced = re.search(r"""(?:name|tool)\s*=\s*["']?([A-Za-z_][\w-]*)""", tag or "", re.I)
            if forced:
                obj = _load_call_body(body) if body.strip() else {}
                if isinstance(obj, dict) and not any(k in obj for k in _NAME_KEYS):
                    body = json.dumps({"name": forced.group(1).lower(), "arguments": obj},
                                      ensure_ascii=False)
            # Gemma 4 часто добавляет пустой <tool_call></tool_call> после нативного
            # вызова. Раньше он превращался в «некорректный JSON» и сжигал раунд.
            if body.strip():
                self.calls_raw.append(body)
        return True

    def _consume_tool_result(self) -> Optional[bool]:
        """Убирает поддельный <tool_result>...</tool_result>, написанный моделью.
        Возвращает True — тег проглочен, False — не tool_result, None — ждём конец тега/блока."""
        m = re.match(r"<\s*tool_result\b[^>]*>", self.buf, re.I)
        if not m:
            return False
        end = self.buf.lower().find("</tool_result>", m.end())
        if end < 0:
            return None
        self.buf = self.buf[end + len("</tool_result>"):]
        self.fake_results += 1
        return True

    def _possible_tool_result(self, tail: str) -> bool:
        m = re.match(r"^<\s*/?\s*([A-Za-z_][\w-]*)", tail)
        if m:
            return "tool_result".startswith(m.group(1).lower())
        return bool(re.match(r"^<\s*/?\s*$", tail))

    def _possible_tag(self, tail: str) -> bool:
        """Похож ли хвост на начало тега известного инструмента (теги приходят по частям)."""
        candidates = ["tool_call"] + sorted(self.names)
        m = re.match(r"^<\s*/?\s*([A-Za-z_][\w-]*)", tail)
        if m:
            # Имя может быть ещё не дописано: <too → tool_call, <web → web_search.
            name = m.group(1).lower()
            return any(c.lower().startswith(name) for c in candidates)
        return bool(re.match(r"^<\s*/?\s*$", tail))


_NAME_KEYS = ("name", "tool", "tool_name")
_ARG_KEYS = ("arguments", "parameters", "args", "input")


def _load_call_body(raw: str):
    """JSON из тела <tool_call>. Допускаем обёртку ```json и лишний текст вокруг объекта:
    локальные модели это регулярно делают, и раньше такой вызов сразу становился ошибкой."""
    body = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", (raw or "").strip())
    try:
        return json.loads(body)
    except ValueError:
        pass
    start, end = body.find("{"), body.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(body[start:end + 1])
        except ValueError:
            return None
    return None


def _call_args(value) -> Optional[dict]:
    """arguments как объект. Строка с JSON тоже принимается (так отдают некоторые серверы).
    None — значение не разбирается в объект."""
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _xmlish_call(raw: str) -> Optional[dict]:
    """Фолбэк для моделей, которые пишут <name>..</name><arguments>..</arguments>
    внутри <tool_call> вместо JSON. Возвращает dict вызова или None."""
    text = str(raw or "").strip()
    if "<" not in text or ">" not in text:
        return None

    def grab(keys: Tuple[str, ...]) -> Optional[str]:
        for key in keys:
            m = re.search(rf"<\s*{re.escape(key)}\s*>(.*?)</\s*{re.escape(key)}\s*>", text, re.S | re.I)
            if m:
                return m.group(1).strip()
        return None

    name = grab(_NAME_KEYS)
    if not name:
        return None
    arg_text = grab(_ARG_KEYS)
    if arg_text is None:
        # Тело похоже на XML-вызов с именем, но без блока аргументов — считаем пустым объектом.
        return {"name": name, "arguments": {}}
    args = _loose_json(arg_text)
    if args is None:
        # Не JSON — оставляем строку как есть; дальше _call_args/валидация решит.
        return {"name": name, "arguments": arg_text}
    return {"name": name, "arguments": args}


_CALL_EXAMPLE = '<tool_call>{"name": "web_search", "arguments": {"queries": ["запрос"]}}</tool_call>'


def _format_error(problem: str, raw: str) -> str:
    """Ошибка для модели: что не так, что она написала и как должно быть."""
    snip = " ".join(str(raw or "").split())[:160]
    return f"{problem}. Получено: {snip!r}. Правильный формат: {_CALL_EXAMPLE}"


def parse_calls(raw_bodies: List[str], known: Optional[Iterable[str]] = None) -> List[dict]:
    """Разбирает тела блоков. Ошибка разбора не роняет ход: вернётся поле error для модели.
    Принимаются синонимы полей (tool/parameters/args), формат OpenAI {"function": {...}}.
    known — имена доступных инструментов: если имени нет, но "type" совпадает с известным
    инструментом (модель пишет {"type": "subagent", ...}), берём его как имя; остальные поля — аргументы."""
    known_set = set(known or ())
    calls: List[dict] = []
    for raw in raw_bodies:
        native = _native_call_body(raw)
        if native is not None:
            raw = native
        obj = _load_call_body(raw)
        xmlish = _xmlish_call(raw)
        if not isinstance(obj, dict):
            obj = xmlish
        if not isinstance(obj, dict):
            calls.append({"name": "", "arguments": {}, "error": _format_error("некорректный JSON в <tool_call>", raw)})
            continue
        for wrapper in ("function", "tool_call"):
            if isinstance(obj.get(wrapper), dict):
                obj = obj[wrapper]
        name = next((obj[k].strip() for k in _NAME_KEYS
                     if isinstance(obj.get(k), str) and obj[k].strip()), "")
        # _load_call_body может принять XML-подобный <name>...</name> как валидный JSON-объект
        # без name/arguments. В этом случае используем XML-фолбэк целиком.
        if not name and isinstance(xmlish, dict):
            obj = xmlish
            name = str(xmlish.get("name") or "").strip()
        type_alias = False
        if not name and isinstance(obj.get("type"), str) and obj["type"].strip() in known_set:
            name = obj["type"].strip()
            type_alias = True
        if not name and known_set:
            # имя ключом: {"web_search": {"queries": [...]}}
            for key, value in obj.items():
                if key in known_set and isinstance(value, dict):
                    name, obj = key, {"arguments": value}
                    break
        if not name:
            calls.append({"name": "", "arguments": {}, "error": _format_error(
                "в <tool_call> нужны поля name и arguments", raw)})
            continue
        key = next((k for k in _ARG_KEYS if k in obj), None)
        if key is None and type_alias:
            args = {k: v for k, v in obj.items() if k != "type"}
        else:
            args = _call_args(obj.get(key) if key else None)
        if args is None:
            calls.append({"name": name, "arguments": {}, "error": "arguments должен быть объектом"})
            continue
        # Нативный tool_call с оборванными аргументами приходит с полем error:
        # такой вызов не должен исполниться с пустыми аргументами.
        passed = obj.get("error") if isinstance(obj, dict) else None
        if isinstance(passed, str) and passed.strip():
            calls.append({"name": name, "arguments": args, "error": passed.strip()[:300]})
            continue
        calls.append({"name": name, "arguments": args})
    return calls


def find_bare_call_span(text: str, known: Iterable[str]) -> Optional[Tuple[dict, int, int]]:
    """(вызов, начало, конец) голого JSON известного инструмента в тексте. None, если нет."""
    known_set = set(known or ())
    if not known_set or not text:
        return None
    # Нативный вызов мог дойти до истории (например, из старого чата) — выполняем его.
    for m in re.finditer(r"<\s*\|\s*tool_call", text, re.I):
        opened = _native_open_at(text, m.start())
        end = _native_call_end(text, m.start() + opened) if opened else -1
        if end < 0:
            continue
        call = parse_calls([text[m.start():end]], known_set)[0]
        if not call.get("error") and call["name"] in known_set:
            return call, m.start(), end
    decoder = json.JSONDecoder()
    pos = 0
    while True:
        start = text.find("{", pos)
        if start < 0:
            return None
        try:
            obj, end = decoder.raw_decode(text, start)
        except ValueError:
            pos = start + 1
            continue
        if isinstance(obj, dict):
            call = parse_calls([json.dumps(obj, ensure_ascii=False)], known_set)[0]
            if not call.get("error") and call["name"] in known_set:
                return call, start, end
        pos = end


def find_bare_call(text: str, known: Iterable[str]) -> Optional[dict]:
    """Голый JSON вызова известного инструмента в тексте ответа (без <tool_call>). None, если нет."""
    found = find_bare_call_span(text, known)
    return found[0] if found else None


def _shrink_json(value, left: int):
    """Сжать структуру так, чтобы json.dumps всё ещё давал валидный JSON."""
    if isinstance(value, str):
        if len(value) <= left:
            return value
        keep = max(0, left - len("…(обрезано)"))
        return value[:keep] + "…(обрезано)"
    if isinstance(value, dict):
        out: Dict[str, object] = {}
        for key, item in value.items():
            k = str(key)[:80]
            out[k] = _shrink_json(item, max(40, left // max(1, len(value))))
            if len(json.dumps(out, ensure_ascii=False)) > left:
                out.pop(k, None)
                out["_truncated"] = True
                break
        return out
    if isinstance(value, list):
        out_list: List[object] = []
        room = min(len(value), 20)
        for item in value[:room]:
            out_list.append(_shrink_json(item, max(40, left // max(1, room))))
            if len(json.dumps(out_list, ensure_ascii=False)) > left:
                out_list.append("…(обрезано)")
                break
        return out_list
    return value


def compact_model_text(text: str, limit: Optional[int] = None) -> Tuple[str, bool, bool]:
    """Уложить результат инструмента в лимит контекста, сохранив валидный JSON.

    Возвращает (текст, было_ли_обрезание, был_ли_JSON). Простое «отрезали хвост»
    превращало ответ MCP в битый JSON и модель тратила раунд на разбор ошибки.
    """
    cap = max(40, int(limit or MAX_RESULT_CHARS))
    body = str(text or "")
    if len(body) <= cap:
        stripped = body.strip()
        is_json = stripped[:1] in ("{", "[")
        return body, False, is_json
    stripped = body.strip()
    if stripped[:1] in ("{", "["):
        try:
            data = json.loads(stripped)
        except ValueError:
            pass
        else:
            shrunk = _shrink_json(data, max(80, cap - 40))
            compact = json.dumps(shrunk, ensure_ascii=False)
            if len(compact) <= cap:
                return compact, True, True
    return body[:cap] + "\n…(обрезано)", True, False


def tool_result_block(name: str, text: str, limit: Optional[int] = None) -> str:
    """Текст результата в том виде, в котором модель его получает.

    limit — пер-вызов потолок контекста (AGENT_* настройки); по умолчанию MAX_RESULT_CHARS."""
    cap = int(limit or MAX_RESULT_CHARS)
    body, _, _ = compact_model_text(text, cap)
    return f'<tool_result name="{name}">\n{body}\n</tool_result>'
