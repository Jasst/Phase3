# -*- coding: utf-8 -*-
"""XTTS-v2 как движок озвучки. Модель подменяется заглушкой: пакет TTS и веса не нужны."""
import io
import sys
import time
import types
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pulsenet.tts import SileroTTS, TTSError, make_tts
from pulsenet.tts_xtts import XTTSTTS, detect_language


class _Synth:
    output_sample_rate = 24000


class _StubXTTS:
    speakers = ["Alpha Voice", "Beta Voice"]

    def __init__(self):
        self.synthesizer = _Synth()
        self.calls = []

    def tts(self, text, language, speaker=None, speaker_wav=None):
        self.calls.append({"text": text, "language": language, "speaker": speaker, "speaker_wav": speaker_wav})
        return [0.0] * 2400                  # 0.1 с тишины при 24 кГц


def _engine(tmp_path, **env):
    eng = XTTSTTS({"TTS_ENABLED": "1", "XTTS_ACCEPT_LICENSE": "1", **env}, str(tmp_path))
    eng._model = _StubXTTS()
    return eng


def _wav_rate(data: bytes) -> int:
    with wave.open(io.BytesIO(data)) as w:
        return w.getframerate()


def _patch_coqui(monkeypatch, cls):
    """Подменить тяжёлый импорт TTS.api, которого может не быть в тестовом окружении."""
    api = types.ModuleType("TTS.api")
    pkg = types.ModuleType("TTS")
    api.TTS = cls
    pkg.api = api
    monkeypatch.setitem(sys.modules, "TTS", pkg)
    monkeypatch.setitem(sys.modules, "TTS.api", api)


# ---------- выбор движка ----------

def test_xtts_is_selectable_and_other_engines_keep_working(tmp_path):
    from pulsenet.tts_piper import PiperTTS

    assert isinstance(make_tts({"TTS_ENGINE": "xtts"}, str(tmp_path)), XTTSTTS)
    assert isinstance(make_tts({}, str(tmp_path)), PiperTTS)       # piper остаётся по умолчанию
    assert isinstance(make_tts({"TTS_ENGINE": "silero"}, str(tmp_path)), SileroTTS)


def test_unknown_engine_fails_fast(tmp_path):
    with pytest.raises(ValueError, match="TTS_ENGINE"):
        make_tts({"TTS_ENGINE": "bogus"}, str(tmp_path))


# ---------- доступ и лицензия ----------

def test_disabled_by_default(tmp_path):
    with pytest.raises(TTSError, match="выключена"):
        XTTSTTS({}, str(tmp_path)).load()


def test_license_must_be_accepted_before_download(tmp_path):
    eng = XTTSTTS({"TTS_ENABLED": "1"}, str(tmp_path))
    with pytest.raises(TTSError, match="XTTS_ACCEPT_LICENSE"):
        eng.load()
    assert "XTTS_ACCEPT_LICENSE" in eng.status()["error"]


def test_transformers_5_is_rejected_clearly(tmp_path, monkeypatch):
    fake = types.ModuleType("transformers")
    fake.__version__ = "5.0.0"
    monkeypatch.setitem(sys.modules, "transformers", fake)
    eng = XTTSTTS({"TTS_ENABLED": "1", "XTTS_ACCEPT_LICENSE": "1",
                   "XTTS_DEVICE": "cpu"}, str(tmp_path))
    with pytest.raises(TTSError, match="transformers 5.x"):
        eng.load()
    assert "transformers>=4.57" in eng.status()["error"]


def test_old_silero_switch_still_enables_engine(tmp_path):
    assert XTTSTTS({"SILERO_ENABLED": "1"}, str(tmp_path)).enabled is True


# ---------- синтез ----------

def test_synth_uses_default_built_in_voice_and_russian(tmp_path):
    eng = _engine(tmp_path)
    data = eng.synth("Привет, как дела?")
    call = eng._model.calls[-1]
    assert call["speaker"] == "Alpha Voice"          # первый голос модели, если не задан
    assert call["language"] == "ru"
    assert _wav_rate(data) == 24000                  # частота берётся из модели, а не из константы


def test_requested_voice_is_used_for_this_request_only(tmp_path):
    eng = _engine(tmp_path, XTTS_SPEAKER="Alpha Voice")
    eng.synth("Тест.", speaker="Beta Voice")
    assert eng._model.calls[-1]["speaker"] == "Beta Voice"
    eng.synth("Тест.")
    assert eng._model.calls[-1]["speaker"] == "Alpha Voice"


def test_unknown_voice_is_value_error_listing_voices(tmp_path):
    eng = _engine(tmp_path)
    with pytest.raises(ValueError, match="Alpha Voice"):
        eng.synth("Тест.", speaker="Nobody")


def test_latin_text_switches_language_to_english(tmp_path):
    eng = _engine(tmp_path)
    eng.synth("This is a short English sentence for the test.")
    assert eng._model.calls[-1]["language"] == "en"


def test_language_detection_keeps_russian_for_mixed_text():
    assert detect_language("Привет, мир") == "ru"
    assert detect_language("Install pip и запусти") == "ru"          # мало латиницы — остаётся ru
    assert detect_language("Just an English sentence here") == "en"


def test_speaker_wav_clones_voice_and_bypasses_list(tmp_path):
    sample = tmp_path / "me.wav"
    sample.write_bytes(b"RIFF")
    eng = _engine(tmp_path, XTTS_SPEAKER_WAV=str(sample))
    eng.synth("Тест.")
    call = eng._model.calls[-1]
    assert call["speaker_wav"] == str(sample) and call["speaker"] is None


def test_missing_speaker_wav_is_reported(tmp_path):
    eng = XTTSTTS({"TTS_ENABLED": "1", "XTTS_ACCEPT_LICENSE": "1",
                   "XTTS_SPEAKER_WAV": str(tmp_path / "nope.wav")}, str(tmp_path))
    with pytest.raises(TTSError, match="образец голоса не найден"):
        eng.load()


def test_model_is_loaded_once_for_several_requests(tmp_path, monkeypatch):
    class _CountingCoqui:
        instances = []

        def __init__(self, _name, progress_bar=False):
            self.__class__.instances.append(self)
            self.speakers = ["Alpha Voice", "Beta Voice"]
            self.synthesizer = _Synth()

        def to(self, _device):
            return self

        def tts(self, text, language, speaker=None, speaker_wav=None):
            return [0.0] * 2400

    _patch_coqui(monkeypatch, _CountingCoqui)
    eng = XTTSTTS({"TTS_ENABLED": "1", "XTTS_ACCEPT_LICENSE": "1",
                   "XTTS_DEVICE": "cpu"}, str(tmp_path))
    eng.synth("Первая фраза.")
    first = id(eng._model)
    eng.synth("Вторая фраза.")
    assert id(eng._model) == first
    assert len(_CountingCoqui.instances) == 1


def test_concurrent_first_requests_load_model_once(tmp_path, monkeypatch):
    class _SlowCoqui:
        instances = []

        def __init__(self, _name, progress_bar=False):
            time.sleep(0.05)                  # окно для гонки между потоками
            self.__class__.instances.append(self)

        def to(self, _device):
            return self

    _patch_coqui(monkeypatch, _SlowCoqui)
    eng = XTTSTTS({"TTS_ENABLED": "1", "XTTS_ACCEPT_LICENSE": "1",
                   "XTTS_DEVICE": "cpu"}, str(tmp_path))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: eng.load(), range(4)))
    assert len(_SlowCoqui.instances) == 1


def test_empty_text_and_code_only_are_rejected(tmp_path):
    with pytest.raises(TTSError, match="нечего озвучивать"):
        _engine(tmp_path).synth("```print(1)```")


def test_synthesis_error_becomes_tts_error(tmp_path):
    eng = _engine(tmp_path)

    def broken(**_kw):
        raise RuntimeError("модель упала")
    eng._model.tts = broken
    with pytest.raises(TTSError, match="модель упала"):
        eng.synth("Тест.")
    assert "модель упала" in eng.status()["error"]


# ---------- статус и голоса ----------

def test_status_lists_configured_voice_before_load(tmp_path):
    eng = XTTSTTS({"TTS_ENABLED": "1", "XTTS_SPEAKER": "Beta Voice"}, str(tmp_path))
    st = eng.status()
    assert st["engine"] == "xtts" and st["ready"] is False and st["available"] is True
    assert st["voices"][0]["id"] == "Beta Voice"


def test_status_lists_model_voices_after_load(tmp_path):
    st = _engine(tmp_path).status()
    assert st["ready"] is True
    assert [v["id"] for v in st["voices"]] == ["Alpha Voice", "Beta Voice"]


def test_menu_lists_builtin_voices_before_model_download(tmp_path):
    eng = XTTSTTS({"TTS_ENABLED": "1"}, str(tmp_path))
    ids = [v["id"] for v in eng.status()["voices"]]
    assert len(ids) > 10 and "Claribel Dervla" in ids and "Ana Florence" in ids
    assert all(v["id"] for v in eng.status()["voices"])   # пустых id, которые нельзя выбрать, нет


def test_stale_silero_voice_is_dropped(tmp_path):
    eng = XTTSTTS({"TTS_ENABLED": "1"}, str(tmp_path))
    assert eng.normalize_speaker("  xenia ") == ""        # голоса Silero больше нет в меню
    assert eng.normalize_speaker("Claribel Dervla") == "Claribel Dervla"
    assert eng.normalize_speaker("") == ""                 # пусто = голос по умолчанию


def test_server_side_voice_check_is_engine_specific(tmp_path):
    assert _engine(tmp_path).normalize_speaker("  Beta Voice ") == "Beta Voice"
    with pytest.raises(ValueError):          # Silero знает свой список и отвечает 400 на чужой голос
        SileroTTS({}, str(tmp_path)).normalize_speaker("nobody")
