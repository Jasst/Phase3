# -*- coding: utf-8 -*-
"""Piper как движок озвучки. Голоса подменяются заглушкой: пакет piper-tts и файлы голосов не нужны."""
import io
import wave

import pytest

from pulsenet.tts import SileroTTS, TTSError, detect_language, make_tts
from pulsenet.tts_piper import PiperTTS, parse_voices


class _Chunk:
    sample_rate = 22050
    sample_width = 2
    sample_channels = 1

    def __init__(self, n=2205):
        self.audio_int16_bytes = b"\x00\x00" * n


class _StubVoice:
    def __init__(self):
        self.texts = []

    def synthesize(self, text):
        self.texts.append(text)
        yield _Chunk()
        yield _Chunk()


def _files(tmp_path, *names):
    out = {}
    for name in names:
        p = tmp_path / f"{name}.onnx"
        p.write_bytes(b"x")
        out[name] = str(p)
    return out


def _engine(tmp_path, voices_env=None, **env):
    f = _files(tmp_path, "irina", "dmitri", "lessac")
    cfg = {"TTS_ENABLED": "1",
           "PIPER_VOICES": voices_env or f"ru-irina={f['irina']};ru-dmitri={f['dmitri']};en-lessac={f['lessac']}",
           **env}
    eng = PiperTTS(cfg, str(tmp_path))
    eng._loaded = {vid: _StubVoice() for vid in eng.voices}
    return eng


def _wav_info(data: bytes):
    with wave.open(io.BytesIO(data)) as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


# ---------- разбор настроек ----------

def test_voices_are_listed_in_order_with_labels(tmp_path):
    eng = _engine(tmp_path)
    assert list(eng.voices) == ["ru-irina", "ru-dmitri", "en-lessac"]
    assert eng.voices["ru-dmitri"]["label"] == "Dmitri (ru)"


def test_legacy_one_voice_per_language_still_works(tmp_path):
    f = _files(tmp_path, "ru", "en")
    voices, err = parse_voices({"PIPER_VOICE_RU": f["ru"], "PIPER_VOICE_EN": f["en"]})
    assert err == "" and list(voices) == ["ru", "en"]


def test_bad_voice_entry_is_reported_not_crashing(tmp_path):
    voices, err = parse_voices({"PIPER_VOICES": "de-hans=/x.onnx"})
    assert voices == {} and "ru- или en-" in err
    voices, err = parse_voices({"PIPER_VOICES": "ru-irina"})
    assert "id=путь" in err


def test_first_voice_of_language_is_default(tmp_path):
    eng = _engine(tmp_path)
    assert eng.default_for("ru") == "ru-irina" and eng.default_for("en") == "en-lessac"


# ---------- выбор движка ----------

def test_piper_is_default_and_silero_still_selectable(tmp_path):
    assert isinstance(make_tts({}, str(tmp_path)), PiperTTS)
    assert isinstance(make_tts({"TTS_ENGINE": "silero"}, str(tmp_path)), SileroTTS)


def test_xtts_is_removed_and_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="piper или silero"):
        make_tts({"TTS_ENGINE": "xtts"}, str(tmp_path))


# ---------- доступ и статус ----------

def test_disabled_by_default(tmp_path):
    with pytest.raises(TTSError, match="выключена"):
        PiperTTS({}, str(tmp_path)).synth("Привет.")


def test_no_voices_configured_gives_clear_message(tmp_path):
    eng = PiperTTS({"TTS_ENABLED": "1"}, str(tmp_path))
    with pytest.raises(TTSError, match="PIPER_VOICES"):
        eng.synth("Привет.")


def test_missing_voice_file_is_reported_in_status(tmp_path):
    eng = PiperTTS({"TTS_ENABLED": "1", "PIPER_VOICES": f"ru-irina={tmp_path / 'нет.onnx'}"}, str(tmp_path))
    st = eng.status()
    assert st["available"] is False and st["missing_voice_files"] == ["ru-irina"]
    with pytest.raises(TTSError, match="файл голоса не найден"):
        eng.synth("Привет.")


def test_status_lists_all_voices_for_the_picker(tmp_path):
    st = _engine(tmp_path).status()
    assert st["engine"] == "piper" and st["available"] is True
    assert [v["id"] for v in st["voices"]] == ["ru-irina", "ru-dmitri", "en-lessac"]


# ---------- синтез ----------

def test_russian_text_uses_default_russian_voice_and_returns_wav(tmp_path):
    eng = _engine(tmp_path)
    data = eng.synth("Привет, как дела?")
    assert eng._loaded["ru-irina"].texts == ["Привет, как дела?"]
    assert eng._loaded["ru-dmitri"].texts == [] and eng._loaded["en-lessac"].texts == []
    assert _wav_info(data) == (22050, 1, 4410)


def test_picked_voice_is_used_for_this_request_only(tmp_path):
    eng = _engine(tmp_path)
    eng.synth("Привет.", speaker="ru-dmitri")
    assert eng._loaded["ru-dmitri"].texts == ["Привет."]
    eng.synth("Привет ещё.")
    assert eng._loaded["ru-irina"].texts == ["Привет ещё."]


def test_latin_text_uses_english_voice(tmp_path):
    eng = _engine(tmp_path)
    eng.synth("This is a short English sentence for the test.")
    assert eng._loaded["en-lessac"].texts and not eng._loaded["ru-irina"].texts


def test_unknown_voice_is_value_error_listing_ids(tmp_path):
    with pytest.raises(ValueError, match="ru-irina"):
        _engine(tmp_path).normalize_speaker("de-hans")


def test_missing_language_falls_back_to_available_voice(tmp_path):
    eng = _engine(tmp_path, voices_env=f"ru-irina={_files(tmp_path, 'only')['only']}")
    eng.synth("This is only English text in this test sentence.")
    assert eng._loaded["ru-irina"].texts


def test_empty_text_and_code_only_are_rejected(tmp_path):
    with pytest.raises(TTSError, match="нечего озвучивать"):
        _engine(tmp_path).synth("```print(1)```")


def test_language_detection_keeps_russian_for_mixed_text():
    assert detect_language("Привет, мир") == "ru"
    assert detect_language("Install pip и запусти") == "ru"
    assert detect_language("Just an English sentence here") == "en"
