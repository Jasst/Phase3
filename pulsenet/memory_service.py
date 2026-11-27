# -*- coding: utf-8 -*-
"""Сборка компонентов памяти из переменных окружения.

Один и тот же код используется и чатом (локальный режим), и MCP-сервером pulsenet-unified,
который в режиме PULSENET_MEMORY=mcp владеет журналом. Эмбеддер и экстрактор поднимаются
там, где хранится память: модель грузится один раз, а не в каждом процессе.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import Callable, Optional, Tuple

from .triplex import ExtractionError, HashEncoder, OpenAIEmbedder, make_extractor, openai_chat_caller
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
            ensure_lmstudio_cpu(model, env)     # чтобы автозагрузка не заняла VRAM
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


def _lmstudio_root(env) -> str:
    """Корень REST API LM Studio из LMSTUDIO_URL / TRIPLEX_LLM_URL."""
    url = (env.get("LMSTUDIO_URL") or env.get("TRIPLEX_LLM_URL") or "").strip().rstrip("/")
    for tail in ("/chat/completions", "/models"):
        if url.endswith(tail):
            url = url[:-len(tail)]
    if not url:
        return ""
    base = url if url.endswith("/v1") else url + "/v1"
    return base[:-3] if base.endswith("/v1") else base


def loaded_model_ids(env) -> Optional[set]:
    """id моделей, уже загруженных в LM Studio. None — сервер недоступен."""
    from .llm import http_json                  # локальный импорт: лёгкий helper без цикла

    root = _lmstudio_root(env)
    if not root:
        return None
    try:
        data = http_json(root + "/api/v0/models", timeout=3)
    except (OSError, ValueError):
        return None
    return {m["id"] for m in data.get("data", [])
            if isinstance(m, dict) and m.get("state") in ("loaded", "loading") and m.get("id")}


def _lms_command() -> str:
    """Путь к lms CLI. REST не умеет --gpu off, поэтому нужен именно CLI."""
    found = shutil.which("lms")
    if found:
        return found
    for cand in (os.path.join(os.path.expanduser("~"), ".lmstudio", "bin", "lms.exe"),
                 os.path.join(os.path.expanduser("~"), ".lmstudio", "bin", "lms")):
        if os.path.isfile(cand):
            return cand
    return "lms"


def ensure_lmstudio_cpu(model: str, env, context: int = 8192) -> bool:
    """Загрузить модель памяти на CPU, если её ещё нет в LM Studio.

    REST /chat/completions грузит модель на GPU сам и не умеет выбрать устройство,
    поэтому при TRIPLEX_LOCAL_DEVICE=cpu используем lms load --gpu off до первого запроса.
    Модель чата этой настройкой не трогается: она грузится как обычно.
    """
    if (env.get("TRIPLEX_LOCAL_DEVICE") or "").strip().lower() != "cpu" or not model:
        return False
    loaded = loaded_model_ids(env)
    if loaded is None or model in loaded:
        return False
    log = logging.getLogger("pulsenet.memory")
    try:
        proc = subprocess.run([_lms_command(), "load", model, "--gpu", "off",
                               "-c", str(context), "-y"],
                              check=False, capture_output=True, text=True, timeout=180)
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        log.warning("lms load %s --gpu off не удался: %s", model, exc)
        return False
    if proc.returncode != 0:
        log.warning("lms load %s --gpu off: rc=%s %s", model, proc.returncode,
                    ((proc.stderr or proc.stdout) or "")[-300:])
        return False
    log.info("модель памяти %s загружена на CPU", model)
    return True


def preload_memory_models(env) -> list:
    """На старте MCP-сервера поднять эмбеддер и экстрактор на CPU,
    если задано TRIPLEX_LOCAL_DEVICE=cpu. Возвращает список загруженных моделей."""
    if (env.get("TRIPLEX_LOCAL_DEVICE") or "").strip().lower() != "cpu":
        return []
    from .triplex.encoder import list_model_ids, pick_embedding_model

    ids = loaded_model_ids(env)
    if ids is None:
        return []
    models = []
    embed_model = (env.get("TRIPLEX_EMBED_MODEL") or "").strip()
    if not embed_model and (env.get("TRIPLEX_EMBED_URL") or "").strip():
        try:
            embed_model = pick_embedding_model(
                list_model_ids((env.get("TRIPLEX_EMBED_URL") or "").strip())) or ""
        except (OSError, ValueError):
            embed_model = ""
    if embed_model:
        models.append(embed_model)
    if (env.get("TRIPLEX_EXTRACT") or "1") != "0":
        llm_model = (env.get("TRIPLEX_LLM_MODEL") or "").strip()
        if llm_model:
            models.append(llm_model)
    loaded = []
    for mid in dict.fromkeys(models):
        try:
            ctx = int(env.get("TRIPLEX_LLM_CONTEXT", "8192"))
        except (TypeError, ValueError):
            ctx = 8192
        if ensure_lmstudio_cpu(mid, env, context=ctx):
            loaded.append(mid)
    return loaded


def loaded_chat_model(env) -> str:
    """Модель, реально загруженная в LM Studio (без эмбеддингов). '' — недоступно.

    Используется и чатом, и MCP-сервером памяти: если пользователь сменил модель
    в LM Studio, факты должны извлекаться ею, а не моделью из .env.
    """
    from .llm import http_json                  # локальный импорт: лёгкий helper без цикла

    root = _lmstudio_root(env)
    if not root:
        return ""
    try:
        data = http_json(root + "/api/v0/models", timeout=3)
    except (OSError, ValueError):
        return ""
    found, vlm, embeddings = [], [], set()
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
        # VLM на text-only запросе часто возвращает пустой content — для JSON не годится.
        (vlm if m.get("type") == "vlm" else found).append((used, m["id"]))
    ids = [mid for _, mid in sorted(found, reverse=True) if mid not in embeddings]
    if ids:
        return ids[0]
    configured = (env.get("TRIPLEX_LLM_MODEL") or "").strip()
    if configured:
        return configured
    ids = [mid for _, mid in sorted(vlm, reverse=True) if mid not in embeddings]
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

    def with_fallback(chosen: str) -> Callable[[str], str]:
        """Чат может идти под своей моделью: экстракция пробует её, затем .env, затем загруженную.

        Так reasoning-модель чата, вернувшая пустой content, не ломает запись памяти —
        факты извлекаются другой, более подходящей для JSON моделью.
        """
        configured = (env.get("TRIPLEX_LLM_MODEL") or "").strip()

        def call(prompt: str) -> str:
            tried, last_err = [], None
            for mid in (chosen, configured, loaded_chat_model(env)):
                mid = (mid or "").strip()
                if not mid or mid in tried:
                    continue
                tried.append(mid)
                try:
                    if mid == configured and (env.get("TRIPLEX_LOCAL_DEVICE") or "").strip().lower() == "cpu":
                        try:
                            ctx = int(env.get("TRIPLEX_LLM_CONTEXT", "8192"))
                        except (TypeError, ValueError):
                            ctx = 8192
                        ensure_lmstudio_cpu(mid, env, context=ctx)
                    return openai_chat_caller(url, mid)(prompt)
                except Exception as exc:  # noqa: BLE001 — чужая модель недоступна или пуста
                    last_err = exc
            raise ExtractionError(
                f"ни одна из моделей ({', '.join(tried) or 'нет'}) не вернула JSON: {last_err}")
        return call

    if explicit:
        return make_extractor(with_fallback(explicit), speaker=speaker)

    chosen = loaded_chat_model(env) or (env.get("TRIPLEX_LLM_MODEL") or "local").strip()
    return make_extractor(with_fallback(chosen), speaker=speaker)
