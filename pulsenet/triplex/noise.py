# -*- coding: utf-8 -*-
"""Фильтр шумных сущностей для графа памяти.

Местоимения, общие слова и обрывки фраз не должны становиться узлами: они
склеивают несвязанные факты и портят активацию (пути идут через «ты», «вещь»,
«на основе моего запроса»). Проверка ставится на всех точках записи:
extract → ingest → assert_triple, плюс разовая чистка уже существующего графа.

Правило здесь простое: в граф попадает только то, что можно процитировать
как отдельный объект. Собственное имя или технический токен после предлога
(«в Dome», «с Gemma 4») — объект; «на основе моего запроса» — нет.
"""
from __future__ import annotations

from typing import Iterable, Optional

NOISE_ENTITY_NAMES = frozenset({
    # русские местоимения и общие слова
    "я", "ты", "вы", "мы", "он", "она", "оно", "они", "это", "то", "эта", "тот", "те",
    "меня", "тебя", "себя", "мне", "тебе", "себе", "мой", "твой", "наш", "ваш",
    "кто", "что", "кого", "чего", "кому", "чему", "ним", "ней", "них",
    "вещь", "вещи", "штука", "штуки", "предмет", "предметы",
    "инструменты", "инструмент", "одежда", "одежду", "объект", "объекты",
    "вопрос", "вопросы", "ответ", "ответы", "запрос", "запросы", "результат",
    "результаты", "пример", "примеры", "случай", "ситуация", "часть",
    # английские двойники — память смешанная, исследователи тоже пишут по-английски
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those",
    "me", "him", "her", "us", "them", "myself", "itself", "thing", "things", "stuff",
    "instruments", "tools", "clothes", "question", "answer", "request", "result", "example",
})

# Обрывок фразы начинается с предлога/союза и НЕ содержит собственного имени
# или технического токена в остатке.
_FRAGMENT_LEAD = frozenset({
    "на", "в", "во", "с", "со", "по", "для", "из", "к", "ко", "от", "о", "об", "у",
    "при", "про", "за", "до", "после", "через", "без", "и", "а", "но", "да",
})
_SUBSTANTIVE = frozenset("-_./:\\")


def _is_substantive(token: str) -> bool:
    """Собственное имя, число или технический идентификатор среди слов обрывка."""
    return (token[:1].isupper() or any(ch.isdigit() for ch in token)
            or any(ch in _SUBSTANTIVE for ch in token))


def noise_reason(name: str, extra_names: Optional[Iterable[str]] = None) -> str:
    """Вернуть причину шума ('' — не шум): empty | generic | fragment.

    ``extra_names`` — проектный стоп-лист, который можно передать без изменения
    глобального набора (например, имена агентов-посредников в конкретном графе).
    """
    raw = (name or "").strip()
    low = raw.lower()
    if not low:
        return "empty"
    extras = {str(x).strip().lower() for x in (extra_names or ()) if str(x).strip()}
    if low in NOISE_ENTITY_NAMES or low in extras:
        return "generic"
    words = raw.split()
    lowered = low.split()
    if len(words) > 1 and lowered[0] in _FRAGMENT_LEAD:
        if not any(_is_substantive(w) for w in words[1:]):
            return "fragment"
    return ""


def is_noise_entity(name: str, extra_names: Optional[Iterable[str]] = None) -> bool:
    """True для имён, которые не должны стать сущностями графа."""
    return bool(noise_reason(name, extra_names))
