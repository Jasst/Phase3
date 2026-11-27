# -*- coding: utf-8 -*-
"""Выбор модели: явный выбор → реально загруженная в LM Studio → .env."""
from types import SimpleNamespace

import pytest

import pulsenet.llm as llm
import pulsenet.memory_service as memory_service
from pulsenet.app import App


class FakeBackend:
    name = "lmstudio"
    loaded = "fake"
    base_url = "http://fake/v1"

    def __init__(self):
        self.ensured = []
        self.unsupported_refresh = False
        self._loaded_for_test = None

    def loaded_model(self, force=False):
        return self._loaded_for_test

    def ensure(self, model, params=None):
        self.ensured.append(model)
        target = model or self.loaded_model() or "fake"
        self.loaded = target
        self._loaded_for_test = target     # LM Studio грузит модель самим запросом
        return self.loaded

    def refresh_loaded(self):
        if self.unsupported_refresh:
            raise AssertionError(" refresh_loaded не должен вызываться при выборе модели")
        return self.loaded_model()

    def list_models(self):
        return ["configured/model", "other-model-in-lmstudio"]


def _app(tmp_path, model="configured/model"):
    env = {"LLM_BACKEND": "lmstudio", "LMSTUDIO_MODEL": model,
           "TRIPLEX_EXTRACT": "0", "TRIPLEX_MEMORY": "1"}
    app = App(env, data_dir=str(tmp_path))
    app.backend = FakeBackend()
    return app


def test_default_is_loaded_model_when_available(tmp_path):
    app = _app(tmp_path)
    app.backend._loaded_for_test = "other-model-in-lmstudio"
    assert app.default_rel() == "other-model-in-lmstudio"
    assert app.resolve_model() == "other-model-in-lmstudio"
    assert app.extraction_model() == "other-model-in-lmstudio"


def test_default_falls_back_to_configured(tmp_path):
    app = _app(tmp_path)
    app.backend.unsupported_refresh = True
    assert app.default_rel() == "configured/model"
    assert app.resolve_model() == "configured/model"
    assert app.extraction_model() == "configured/model"


def test_memory_operations_do_not_change_chosen_model(tmp_path):
    app = _app(tmp_path)
    app.memory_backend = "mcp"       # в MCP-режиме модель передаётся явно
    app.active_model = "chat/model"
    seen = []

    def fake_ingest(text, **kwargs):
        seen.append(kwargs)
        return {"status": "ok"}

    app.memory.ingest = fake_ingest

    app.ingest("текст", model="explicit/for-this-call")
    assert app.active_model == "chat/model"          # глобальный выбор не тронут
    assert seen[-1]["model"] == "explicit/for-this-call"

    app.ingest("ещё текст")
    assert seen[-1]["model"] == "chat/model"         # память идёт моделью чата
    assert app.backend.ensured == []                 # запись не загружает модель


def test_status_shows_loaded_and_fact_model(tmp_path):
    app = _app(tmp_path)
    app.backend._loaded_for_test = "chat/model"
    app.backend.loaded = "chat/model"
    app.memory.stats = lambda: {}
    status = app.status()
    assert status["model"] == "chat/model"
    assert status["loaded_model"] == "chat/model"
    assert status["fact_model"] == "chat/model"


def test_lmstudio_list_models_is_read_only(monkeypatch):
    backend = llm.LMStudioBackend({"LMSTUDIO_URL": "http://fake/v1",
                                   "LMSTUDIO_MODEL": "configured/model"})
    backend.loaded = "configured/model"

    def fake_http_json(url, timeout=5):
        if url.endswith("/api/v0/models"):
            return {"data": [
                {"id": "other-model-in-lmstudio", "type": "llm", "state": "loaded"},
                {"id": "embedding", "type": "embeddings", "state": "loaded"},
            ]}
        return {"data": [{"id": "configured/model"}, {"id": "other-model-in-lmstudio"},
                         {"id": "embedding"}]}

    monkeypatch.setattr(llm, "http_json", fake_http_json)
    assert backend.list_models() == ["configured/model", "other-model-in-lmstudio"]
    assert backend.loaded == "configured/model"       # список не меняет выбор


def test_chat_uses_explicit_model_and_facts_follow_it(tmp_path):
    app = _app(tmp_path)

    def script(model, messages, on_delta, cancelled, **kwargs):
        app.seen_models.append(model)
        on_delta("token", "ok")

    app.seen_models = []
    app.backend.stream = script
    app.chat("первый", "c1", lambda kind, payload: None, model="user/choice")
    assert app.seen_models[-1] == "user/choice"
    assert app.active_model == "user/choice"
    assert app.extraction_model() == "user/choice"    # LM Studio теперь держит user/choice

    # Второй ход без model тоже идёт загруженной моделью, а не «первой из .env».
    app.chat("второй", "c1", lambda kind, payload: None)
    assert app.seen_models[-1] == "user/choice"
    assert app.extraction_model() == "user/choice"


def test_lmstudio_ensure_picks_loaded_model_without_choice(monkeypatch):
    backend = llm.LMStudioBackend({"LMSTUDIO_URL": "http://fake/v1", "LMSTUDIO_MODEL": ""})
    monkeypatch.setattr(backend, "_loaded_ids", lambda: ["other-model-in-lmstudio"])
    monkeypatch.setattr(backend, "_known_ids", lambda: ["other-model-in-lmstudio", "user/choice"])
    assert backend.ensure("") == "other-model-in-lmstudio"
    assert backend.loaded == "other-model-in-lmstudio"

    # Явный выбор всё равно имеет приоритет.
    assert backend.ensure("user/choice") == "user/choice"
    assert backend.loaded == "user/choice"


def test_lmstudio_ensure_fails_when_nothing_loaded_and_no_default(monkeypatch):
    backend = llm.LMStudioBackend({"LMSTUDIO_URL": "http://fake/v1", "LMSTUDIO_MODEL": ""})
    monkeypatch.setattr(backend, "_loaded_ids", lambda: [])
    with pytest.raises(llm.BackendError, match="модель не выбрана"):
        backend.ensure("")
    assert backend.loaded is None


def test_llamacpp_external_does_not_invent_model(monkeypatch):
    backend = llm.LlamaCppBackend({"LLAMACPP_DEFAULT_MODEL": ""})
    backend.healthy = lambda: True
    backend.proc = None
    backend.loaded = None
    with pytest.raises(llm.BackendError, match="модель не выбрана"):
        backend.ensure("")
    assert backend.loaded is None

    assert backend.ensure("user/choice") == "user/choice"
    assert backend.loaded == "user/choice"
    assert backend.external is True


def test_status_uses_loaded_model_not_configured(tmp_path):
    app = _app(tmp_path)
    app.backend._loaded_for_test = "other-model-in-lmstudio"
    app.backend.loaded = "other-model-in-lmstudio"
    app.memory.stats = lambda: {}
    assert app.status()["model"] == "other-model-in-lmstudio"
    assert app.status()["fact_model"] == "other-model-in-lmstudio"


def test_loaded_chat_model_skips_embeddings_and_uses_last_used(monkeypatch):
    def fake_http_json(url, timeout=10):
        if url.endswith("/api/v0/models"):
            return {"data": [
                {"id": "emb", "type": "embeddings", "state": "loaded", "last_used": 99},
                {"id": "old", "type": "llm", "state": "loaded", "last_used": 1},
                {"id": "new", "type": "llm", "state": "loaded", "last_used": 2},
            ]}
        raise AssertionError(url)

    monkeypatch.setattr(llm, "http_json", fake_http_json)
    env = {"LMSTUDIO_URL": "http://localhost:1234/v1", "TRIPLEX_LLM_MODEL": "env-model"}
    assert memory_service.loaded_chat_model(env) == "new"


def test_only_vlm_loaded_uses_configured_text_model(monkeypatch):
    def fake_http_json(url, timeout=10):
        return {"data": [
            {"id": "emb", "type": "embeddings", "state": "loaded", "last_used": 99},
            {"id": "vlm", "type": "vlm", "state": "loaded", "last_used": 50},
        ]}

    monkeypatch.setattr(llm, "http_json", fake_http_json)
    env = {"LMSTUDIO_URL": "http://localhost:1234/v1", "TRIPLEX_LLM_MODEL": "text-model"}
    assert memory_service.loaded_chat_model(env) == "text-model"


def test_ensure_lmstudio_cpu_loads_missing_model(monkeypatch):
    calls = []
    monkeypatch.setattr(memory_service, "loaded_model_ids", lambda env: set())
    monkeypatch.setattr(memory_service, "_lms_command", lambda: "lms")
    monkeypatch.setattr(memory_service.subprocess, "run",
                        lambda *a, **kw: calls.append((a[0], kw)) or
                        SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert memory_service.ensure_lmstudio_cpu("emb", {"TRIPLEX_LOCAL_DEVICE": "cpu"}, context=8192)
    assert calls[0][0] == ["lms", "load", "emb", "--gpu", "off", "-c", "8192", "-y"]


def test_ensure_lmstudio_cpu_skips_loaded_and_gpu_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(memory_service, "loaded_model_ids", lambda env: {"emb"})
    monkeypatch.setattr(memory_service.subprocess, "run",
                        lambda *a, **kw: calls.append(a[0]))
    assert not memory_service.ensure_lmstudio_cpu("emb", {"TRIPLEX_LOCAL_DEVICE": "cpu"})
    assert not memory_service.ensure_lmstudio_cpu("other", {"TRIPLEX_LOCAL_DEVICE": "gpu"})
    assert calls == []


def test_build_extractor_preloads_configured_model_on_cpu(monkeypatch):
    seen = []
    monkeypatch.setattr(memory_service, "ensure_lmstudio_cpu",
                        lambda mid, env, context=8192: seen.append(mid) or True)
    monkeypatch.setattr(memory_service, "loaded_chat_model", lambda env: "")
    monkeypatch.setattr(memory_service, "openai_chat_caller",
                        lambda url, model: (lambda prompt: '{"facts": [], "entities": []}'))
    env = {"TRIPLEX_LLM_URL": "http://llm/v1", "TRIPLEX_LLM_MODEL": "extract-model",
           "TRIPLEX_LOCAL_DEVICE": "cpu"}
    memory_service.build_extractor(env)("текст")
    assert seen == ["extract-model"]


def test_preload_memory_models_loads_embedder_and_extractor(monkeypatch):
    seen = []
    monkeypatch.setattr(memory_service, "loaded_model_ids", lambda env: {"other"})
    monkeypatch.setattr(memory_service, "ensure_lmstudio_cpu",
                        lambda mid, env, context=8192: seen.append(mid) or True)
    env = {"TRIPLEX_LOCAL_DEVICE": "cpu", "TRIPLEX_EMBED_MODEL": "bge-m3",
           "TRIPLEX_LLM_MODEL": "gemma-4-e4b"}
    assert memory_service.preload_memory_models(env) == ["bge-m3", "gemma-4-e4b"]
    assert seen == ["bge-m3", "gemma-4-e4b"]


def test_preload_memory_models_noop_without_cpu_mode(monkeypatch):
    monkeypatch.setattr(memory_service, "ensure_lmstudio_cpu",
                        lambda *a, **kw: pytest.fail("не должно грузиться"))
    assert memory_service.preload_memory_models({"TRIPLEX_EMBED_MODEL": "bge"}) == []
