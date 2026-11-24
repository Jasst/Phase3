# -*- coding: utf-8 -*-
"""Сборка компонентов памяти из переменных окружения.

Один и тот же код используется и чатом (локальный режим), и MCP-сервером pulsenet-unified,
который в режиме PULSENET_MEMORY=mcp владеет журналом. Эмбеддер и экстрактор поднимаются
там, где хранится память: модель грузится один раз, а не в каждом процессе.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

from .triplex import HashEncoder, OpenAIEmbedder, make_extractor, openai_chat_caller
from .triplex.embed_server import EmbeddingServer, EmbeddingServerError
from .triplex.encoder import CachedEncoder, default_prefixes, list_model_ids, pick_embedding_model, probe_dim


def build_encoder(env, data_dir: str, log_error: Callable[[str, BaseException], None]) -> Tuple[object, Optional[EmbeddingServer], str]:
    """Эмбеддер: (encoder, embed_server, model_name).

    bge-m3 грузится один раз — отдельным процессом (TRIPLEX_EMBED_MODEL_PATH) или уже запущенным
    сервером (TRIPLEX_EMBED_URL). Иначе — офлайн-хеш. Сбой запуска не роняет сервис."""
    url = (env.get("TRIPLEX_EMBED_URL") or "").strip()
    model_path = (env.get("TRIPLEX_EMBED_MODEL_PATH") or "").strip()
    model = (env.get("TRIPLEX_EMBED_MODEL") or "").strip()          # пусто — определить автоматически
    dim_env = (env.get("TRIPLEX_EMBED_DIM") or "").strip()
    pooling = (env.get("TRIPLEX_EMBED_POOLING") or "cls").strip()
    server = None
    if model_path and env.get("TRIPLEX_EMBED_AUTOSTART", "1") != "0":
        server = EmbeddingServer(
            (env.get("TRIPLEX_EMBED_BIN") or "").strip() or (env.get("LLAMACPP_SERVER_BIN") or "llama-server").strip(),
            model_path, port=int(env.get("TRIPLEX_EMBED_PORT", "8091")), pooling=pooling,
            log_path=os.path.join(data_dir, "embed-server.log"))
        try:
            url = server.start()
        except Exception as exc:              # noqa: BLE001
            log_error("эмбеддер (запуск)", exc)
            url = ""
    if url:
        try:
            if not model:
                model = pick_embedding_model(list_model_ids(url)) or ""
                if not model:
                    raise EmbeddingServerError(f"на {url} нет модели эмбеддингов: задайте TRIPLEX_EMBED_MODEL")
            dim = probe_dim(url, model)                      # размерность — из ответа модели
            if dim_env and int(dim_env) != dim:
                log_error("эмбеддер (размерность)",
                          ValueError(f"TRIPLEX_EMBED_DIM={dim_env}, а модель {model!r} даёт {dim}: беру {dim}"))
            auto_doc, auto_query = default_prefixes(model)
            doc_prefix = env.get("TRIPLEX_EMBED_DOC_PREFIX", auto_doc)
            query_prefix = env.get("TRIPLEX_EMBED_QUERY_PREFIX", auto_query)
            inner = OpenAIEmbedder(url, model, dim, doc_prefix=doc_prefix, query_prefix=query_prefix)
            enc = CachedEncoder(inner, os.path.join(data_dir, "triplex", "embed_cache.jsonl"),
                                tag=f"{model}|{dim}|{pooling}|{doc_prefix}|{query_prefix}")
            return enc, server, model
        except Exception as exc:              # noqa: BLE001
            log_error("эмбеддер (подключение)", exc)
    return HashEncoder(512), server, ""


def loaded_chat_model(env) -> str:
    """Модель, реально загруженная в LM Studio (без эмбеддингов). '' — недоступно.

    Используется и чатом, и MCP-сервером памяти: если пользователь сменил модель
    в LM Studio, факты должны извлекаться ею, а не моделью из .env.
    """
    from .llm import http_json                  # локальный импорт: лёгкий helper без цикла

    url = (env.get("LMSTUDIO_URL") or env.get("TRIPLEX_LLM_URL") or "").strip().rstrip("/")
    for tail in ("/chat/completions", "/models"):
        if url.endswith(tail):
            url = url[:-len(tail)]
    if not url:
        return ""
    base = url if url.endswith("/v1") else url + "/v1"
    root = base[:-3] if base.endswith("/v1") else base
    try:
        data = http_json(root + "/api/v0/models", timeout=3)
    except (OSError, ValueError):
        return ""
    found, embeddings = [], set()
    for m in data.get("data", []):
        if not isinstance(m, dict) or not m.get("id"):
            continue
        if m.get("type") == "embeddings":
            embeddings.add(m["id"])
            continue
        if m.get("state") not in ("loaded", "loading"):
            continue
        try:
            used = float(m.get("last_used") or 0)
        except (TypeError, ValueError):
            used = 0.0
        found.append((used, m["id"]))
    ids = [mid for _, mid in sorted(found, reverse=True) if mid not in embeddings]
    return ids[0] if ids else ""


def build_extractor(env, model: str = "") -> Optional[Callable[[str], dict]]:
    """Экстрактор фактов через OpenAI-совместимый сервер (TRIPLEX_LLM_URL). Без адреса — None:
    документы сохраняются, факты не извлекаются.

    model — явно выбранная модель (например, из веб-чата). Если пусто, на каждый вызов
    берётся реально загруженная в LM Studio модель, и только при её отсутствии —
    TRIPLEX_LLM_MODEL из .env. Так факты идут моделью чата без второй загрузки.
    """
    if (env.get("TRIPLEX_EXTRACT") or "1") == "0":
        return None
    url = (env.get("TRIPLEX_LLM_URL") or "").strip()
    if not url:
        return None
    explicit = (model or "").strip()
    speaker = (env.get("TRIPLEX_USER_NAME") or "").strip() or None
    if explicit:
        return make_extractor(openai_chat_caller(url, explicit), speaker=speaker)

    def call(prompt: str) -> str:
        chosen = loaded_chat_model(env) or (env.get("TRIPLEX_LLM_MODEL") or "local").strip()
        return openai_chat_caller(url, chosen)(prompt)

    return make_extractor(call, speaker=speaker)
