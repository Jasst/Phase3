# -*- coding: utf-8 -*-
"""Озвучка через Coqui XTTS-v2: одна мультиязычная модель (17 языков, среди них русский), CPU или GPU.

Голос:
  XTTS_SPEAKER      — встроенный голос модели (имя из списка модели; пусто — первый в списке);
  XTTS_SPEAKER_WAV  — образец голоса (wav, 6–30 с): модель копирует тембр. Если задан, голос из списка не нужен.
Язык: XTTS_LANGUAGE (по умолчанию ru). Если текст в основном латиницей, берётся en.

Лицензия модели — CPML (некоммерческая). Перед первой загрузкой задайте XTTS_ACCEPT_LICENSE=1.
Пакет: coqui-tts (см. requirements-tts.txt). Модель (несколько гигабайт) скачивается при первом запросе.
"""
from __future__ import annotations

import os
import threading

import numpy as np

from .tts import MAX_SEGMENT, TTSError, clean_for_speech, pcm_to_wav

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = "ru"


def detect_language(text: str, default: str = DEFAULT_LANGUAGE) -> str:
    """Грубая проверка: если латинских букв заметно больше кириллических — английский, иначе default."""
    cyr = sum(1 for ch in text if "а" <= ch.lower() <= "я" or ch in "ёЁ")
    lat = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return "en" if lat > 2 * cyr and lat >= 10 else default


def _flag(env, key: str) -> bool:
    return str(env.get(key, "")).strip() == "1"


class XTTSTTS:
    """Тот же интерфейс, что SileroTTS: status, load, synth, reset_error, normalize_speaker."""

    engine = "xtts"

    def __init__(self, env: dict, data_dir: str):
        self.enabled = _flag(env, "TTS_ENABLED") or _flag(env, "SILERO_ENABLED")
        self.model_name = (env.get("XTTS_MODEL") or MODEL_NAME).strip()
        self.speaker = (env.get("XTTS_SPEAKER") or "").strip()
        self.speaker_wav = (env.get("XTTS_SPEAKER_WAV") or "").strip()
        self.language = (env.get("XTTS_LANGUAGE") or DEFAULT_LANGUAGE).strip().lower()
        self.device = (env.get("XTTS_DEVICE") or "").strip()    # пусто — cuda, если доступна, иначе cpu
        self.accept_license = _flag(env, "XTTS_ACCEPT_LICENSE")
        self.device_used = ""
        self._model = None
        self._lock = threading.Lock()
        self.error = ""

    # ----- состояние -----

    @property
    def ready(self) -> bool:
        return self._model is not None

    def reset_error(self) -> None:
        self.error = ""

    def normalize_speaker(self, value=None) -> str:
        """Голос из запроса. Список голосов модели известен только после загрузки, поэтому проверка — в synth."""
        return str(value or "").strip()

    def _voices(self) -> list:
        if self.speaker_wav:
            return [{"id": "", "label": "Образец голоса", "description": os.path.basename(self.speaker_wav),
                     "preview": ""}]
        if self._model is not None:
            names = list(getattr(self._model, "speakers", None) or [])
            return [{"id": n, "label": n, "description": "встроенный голос XTTS", "preview": ""} for n in names]
        # до загрузки список неизвестен: показываем заданный голос или подсказку
        label = self.speaker or "первый голос модели"
        return [{"id": self.speaker, "label": label, "description": "станет доступен после загрузки модели",
                 "preview": ""}]

    def status(self) -> dict:
        return {"enabled": self.enabled, "ready": self.ready, "engine": self.engine,
                "speaker": self.speaker, "device": self.device_used or self.device or "auto",
                "model": self.model_name, "language": self.language,
                "voices": self._voices(),
                "available": self.enabled and not self.error,
                "error": self.error}

    # ----- загрузка -----

    def load(self):
        """Модель XTTS или TTSError. Повторная попытка возможна после ошибки."""
        if not self.enabled:
            raise TTSError("озвучка выключена (TTS_ENABLED=1 включает её)")
        with self._lock:
            if self._model is not None:
                return self._model
            if not self.accept_license:
                msg = ("лицензия XTTS-v2 (CPML, некоммерческая). Чтобы разрешить загрузку, "
                       "задайте XTTS_ACCEPT_LICENSE=1 в .env")
                self.error = msg
                raise TTSError(msg)
            if self.speaker_wav and not os.path.isfile(self.speaker_wav):
                msg = f"образец голоса не найден: {self.speaker_wav}"
                self.error = msg
                raise TTSError(msg)
            try:
                os.environ.setdefault("COQUI_TOS_AGREED", "1")   # подтверждение лицензии для библиотеки
                import torch                                      # тяжёлые импорты только при первом запросе
                from TTS.api import TTS
                device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                model = TTS(self.model_name, progress_bar=False).to(device)
                self._model = model
                self.device_used = device
                self.error = ""
                return model
            except Exception as exc:              # noqa: BLE001 — любая ошибка загрузки → статус, не падение
                self.error = f"не удалось загрузить XTTS: {exc}"
                raise TTSError(self.error) from exc

    # ----- синтез -----

    def synth(self, text: str, speaker: str | None = None) -> bytes:
        """WAV для одной фразы. Голос из запроса не меняет голос по умолчанию на сервере."""
        clean = clean_for_speech(text)
        if not clean:
            raise TTSError("нечего озвучивать")
        model = self.load()
        kwargs = {"text": clean[:MAX_SEGMENT * 2], "language": detect_language(clean, self.language)}
        if self.speaker_wav:
            kwargs["speaker_wav"] = self.speaker_wav
        else:
            names = list(getattr(model, "speakers", None) or [])
            voice = (speaker or self.speaker or (names[0] if names else "")).strip()
            if voice not in names:
                raise ValueError(f"неизвестный голос {voice!r}. Доступны: {', '.join(names[:12])}…")
            kwargs["speaker"] = voice
        with self._lock:
            try:
                wav = model.tts(**kwargs)
            except Exception as exc:              # noqa: BLE001 — ошибка синтеза → 503, а не 500
                self.error = f"не удалось синтезировать речь: {exc}"
                raise TTSError(self.error) from exc
        rate = int(model.synthesizer.output_sample_rate)
        return pcm_to_wav(np.asarray(wav, dtype=np.float32), rate)
