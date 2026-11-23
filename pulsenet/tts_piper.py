# -*- coding: utf-8 -*-
"""Озвучка через Piper: локально, без torch и FFmpeg (только onnxruntime).

Голоса — файлы .onnx (рядом должен лежать .onnx.json). Задаются в .env двумя способами:

  PIPER_VOICES=ru-irina=C:\\voices\\ru_RU-irina-medium.onnx;ru-dmitri=C:\\voices\\ru_RU-dmitri-medium.onnx;en-lessac=C:\\voices\\en_US-lessac-medium.onnx

  Формат: id=путь, голоса разделяются «;». id начинается с языка (ru- или en-). В интерфейсе голоса
  показываются в этом порядке. Первый голос каждого языка — голос по умолчанию для этого языка.

  PIPER_VOICE_RU / PIPER_VOICE_EN — один голос на язык (совместимость со старыми настройками).

Язык фразы определяется по тексту; голос из интерфейса переопределяет язык для одного запроса.
"""
from __future__ import annotations

import io
import os
import threading
import wave

from .tts import MAX_SEGMENT, TTSError, clean_for_speech, detect_language

LANGS = ("ru", "en")
LANG_LABELS = {"ru": "Русский", "en": "English"}


def parse_voices(env: dict):
    """Возвращает (голоса, ошибка). Голоса: {id: {"lang", "path", "label"}} в порядке записи."""
    voices: dict = {}
    for lang in LANGS:                                    # совместимость: PIPER_VOICE_RU / PIPER_VOICE_EN
        path = (env.get(f"PIPER_VOICE_{lang.upper()}") or "").strip()
        if path:
            voices[lang] = {"lang": lang, "path": path, "label": LANG_LABELS[lang]}
    raw = (env.get("PIPER_VOICES") or "").strip()
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            return voices, f"PIPER_VOICES: элемент {item!r} без «=» (нужно id=путь)"
        vid, path = (part.strip() for part in item.split("=", 1))
        vid = vid.lower()
        lang = vid.split("-", 1)[0]
        if lang not in LANGS:
            return voices, f"PIPER_VOICES: id {vid!r} должен начинаться с ru- или en-"
        suffix = vid.split("-", 1)[1] if "-" in vid else ""
        label = f"{suffix.capitalize()} ({lang})" if suffix else LANG_LABELS[lang]
        voices[vid] = {"lang": lang, "path": path, "label": label}
    return voices, ""


class PiperTTS:
    """Тот же интерфейс, что SileroTTS: status, load, synth, reset_error, normalize_speaker."""

    engine = "piper"

    def __init__(self, env: dict, data_dir: str):
        self.enabled = str(env.get("TTS_ENABLED", "")).strip() == "1"
        self.voices, config_error = parse_voices(env)
        self.speaker = ""                       # пусто — голос выбирается по языку
        self._loaded: dict = {}                 # id голоса → загруженный PiperVoice
        self._lock = threading.Lock()
        self.error = config_error

    # ----- состояние -----

    @property
    def ready(self) -> bool:
        return bool(self._loaded)

    def reset_error(self) -> None:
        self.error = ""

    def normalize_speaker(self, value=None) -> str:
        """Голос из запроса: id из PIPER_VOICES (или ru/en). Пусто — выбор по языку."""
        raw = str(value or "").strip().lower()
        if not raw:
            return ""
        if raw not in self.voices:
            raise ValueError(f"голос {raw!r} не настроен. Доступны: {', '.join(self.voices) or 'нет'}")
        return raw

    def default_for(self, lang: str):
        """Первый голос указанного языка (или None)."""
        return next((vid for vid, v in self.voices.items() if v["lang"] == lang), None)

    def _voice_list(self) -> list:
        return [{"id": vid, "label": v["label"], "description": os.path.basename(v["path"]), "preview": ""}
                for vid, v in self.voices.items()]

    def status(self) -> dict:
        missing = [vid for vid, v in self.voices.items() if not os.path.isfile(v["path"])]
        return {"enabled": self.enabled, "ready": self.ready, "engine": self.engine,
                "speaker": self.speaker, "device": "cpu", "voices": self._voice_list(),
                "missing_voice_files": missing,
                "available": self.enabled and not self.error and bool(self.voices) and not missing,
                "error": self.error}

    # ----- загрузка -----

    def _load_voice(self, vid: str):
        with self._lock:
            if vid in self._loaded:
                return self._loaded[vid]
            path = self.voices[vid]["path"]
            if not os.path.isfile(path):
                msg = f"файл голоса не найден: {path}"
                self.error = msg
                raise TTSError(msg)
            try:
                from piper import PiperVoice     # импорт только при первом запросе
                voice = PiperVoice.load(path)
            except Exception as exc:              # noqa: BLE001 — любая ошибка загрузки → статус, не падение
                self.error = f"не удалось загрузить голос Piper ({vid}): {exc}"
                raise TTSError(self.error) from exc
            self._loaded[vid] = voice
            self.error = ""
            return voice

    def load(self):
        """Загружает голос по умолчанию или бросает TTSError."""
        if not self.enabled:
            raise TTSError("озвучка выключена (TTS_ENABLED=1 включает её)")
        vid = self.default_for("ru") or next(iter(self.voices), None)
        if vid is None:
            raise TTSError(self.error or "голоса Piper не настроены: задайте PIPER_VOICES в .env")
        return self._load_voice(vid)

    # ----- синтез -----

    def synth(self, text: str, speaker: str | None = None) -> bytes:
        """WAV для одной фразы. Голос из запроса не меняет настройки на сервере."""
        clean = clean_for_speech(text)
        if not clean:
            raise TTSError("нечего озвучивать")
        if not self.enabled:
            raise TTSError("озвучка выключена (TTS_ENABLED=1 включает её)")
        if not self.voices:
            raise TTSError(self.error or "голоса Piper не настроены: задайте PIPER_VOICES в .env")
        vid = self.normalize_speaker(speaker)
        if not vid:
            lang = detect_language(clean, "ru")
            vid = self.default_for(lang) or next(iter(self.voices))
        voice = self._load_voice(vid)
        buf = io.BytesIO()
        try:
            with wave.open(buf, "wb") as w:
                started = False
                for chunk in voice.synthesize(clean[:MAX_SEGMENT * 2]):
                    if not started:
                        w.setnchannels(chunk.sample_channels)
                        w.setsampwidth(chunk.sample_width)
                        w.setframerate(chunk.sample_rate)
                        started = True
                    w.writeframes(chunk.audio_int16_bytes)
                if not started:
                    raise TTSError("Piper не вернул звук для этого текста")
        except TTSError:
            raise
        except Exception as exc:                  # noqa: BLE001 — ошибка синтеза → 503, а не 500
            self.error = f"не удалось синтезировать речь: {exc}"
            raise TTSError(self.error) from exc
        return buf.getvalue()
