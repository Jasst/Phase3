# -*- coding: utf-8 -*-
"""Baseline метрик Phase4+ на синтетическом графе с известным правильным путём.

Запуск:  python eval/eval_phase4.py
Печатает JSON с шестью метриками из docs/PHASE4.md. Это не CI-гейт, а
воспроизводимая точка отсчёта: число вместо «кажется умнее».
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pulsenet.triplex import HashEncoder, TriplexMemory  # noqa: E402

FACTS = [
    ("Алиса", "работает в", "Lab"),
    ("Lab", "находится в", "Москва"),
    ("Алиса", "пьёт", "чай"),
    ("чай", "пьют в", "утро"),
]
CASES = [
    {"query": "Алиса", "goal": "где работает Алиса", "target": "Алиса работает в Lab",
     "wrong": ["Алиса пьёт чай"]},
    {"query": "Алиса", "goal": "что пьёт Алиса", "target": "Алиса пьёт чай",
     "wrong": ["Алиса работает в Lab"]},
    {"query": "Алиса", "goal": "где находится Lab", "target": "Lab находится в Москва",
     "wrong": ["чай пьют в утро"]},
]
TARGETS = {
    "path_hit@5": 0.8,
    "goal_sensitivity": 0.66,
    "false_activation_rate": 0.2,
    "feedback_gain": 0.0,
    "scope_leak": 0.0,
    "determinism": 1.0,
}


def build(root: str):
    mem = TriplexMemory(root, encoder=HashEncoder(256))
    doc = mem.ingest("Алиса работает в Lab. Lab находится в Москве. Алиса пьёт чай. Чай пьют утром.")
    src = {"doc_id": doc["doc_id"], "span": [0, 10]}
    ids = {}
    for s, r, o in FACTS:
        res = mem.assert_triple(s, r, o, source=src)
        ids[res["text"]] = res["fact_id"]
    return mem, ids


def activation_of(th, fid: str) -> float:
    return float((th.get("activation") or {}).get(fid) or 0.0)


def main() -> int:
    root = tempfile.mkdtemp(prefix="phase4_eval_")
    mem, ids = build(root)

    thoughts = {}
    for i, case in enumerate(CASES):
        thoughts[i] = mem.think(case["query"], goal=case["goal"], steps=2,
                                scope=f"eval:case{i}")

    hits = sum(ids[c["target"]] in {f["fact_id"] for f in t["facts"][:5]}
               for c, t in zip(CASES, thoughts.values()))
    # Считаем по лидеру (top-1): в крошечном графе даже top-3 содержит
    # связные, но не отвечающие на вопрос звенья — это не ошибка, а контекст.
    wrong = sum(ids[w] == t["path"][-1]["fact_id"]
                for c, t in zip(CASES, thoughts.values()) for w in c["wrong"])

    # Чувствительность к цели: у одного запроса разные цели дают разные концы пути.
    pairs = [(a, b) for a in range(len(CASES)) for b in range(a + 1, len(CASES))
             if CASES[a]["query"] == CASES[b]["query"]]
    sensitive = sum(thoughts[a]["path"][-1]["fact_id"] != thoughts[b]["path"][-1]["fact_id"]
                    for a, b in pairs)
    sensitivity = sensitive / max(1, len(pairs))

    # Подкрепление меряем по boost в журнале: повторный recall поднимает access_count
    # (I10) у всех top-фактов и зашумлял бы сравнение активации.
    probe = CASES[0]
    fid = ids[probe["target"]]
    before = float(mem.journal.facts[fid].boost)
    tid = mem.think(probe["query"], goal=probe["goal"], steps=2, scope="eval:gain0")["thought_id"]
    mem.feedback([fid], 1.0, thought_id=tid, scope="eval:gain0")
    after_plus = float(mem.journal.facts[fid].boost)
    mem.feedback([fid], -1.0, thought_id=tid, scope="eval:gain0")
    after_minus = float(mem.journal.facts[fid].boost)

    foreign = mem.journal.assert_fact("ёж", "спит", "нора",
                                      source={"doc_id": "d:1", "span": [0, 1]}).fact_id
    leak_test = mem.feedback([foreign], 1.0, scope="eval:case0")
    leaked = 0 if leak_test["status"] == "rejected" else 1
    leaked += len(mem.thoughts.recent(10, scope="eval:other"))

    again = mem.think(probe["query"], goal=probe["goal"], steps=2, scope="eval:determinism")
    deterministic = float([p["ids"] for p in again["paths"]] ==
                          [p["ids"] for p in mem.think(probe["query"], goal=probe["goal"], steps=2,
                                                       scope="eval:determinism2")["paths"]])

    metrics = {
        "path_hit@5": round(hits / len(CASES), 3),
        "goal_sensitivity": round(sensitivity, 3),
        "false_activation_rate": round(wrong / len(CASES), 3),
        "feedback_gain": {
            "after_plus": round(after_plus - before, 4),
            "after_minus": round(after_minus - after_plus, 4),
            "ok": after_plus > before and after_minus < after_plus,
        },  # дельта boost: +1 подкрепляет, -1 ослабляет
        "scope_leak": leaked,
        "determinism": deterministic,
    }
    targets_ok = {
        "path_hit@5": metrics["path_hit@5"] >= TARGETS["path_hit@5"],
        "goal_sensitivity": metrics["goal_sensitivity"] >= TARGETS["goal_sensitivity"],
        "false_activation_rate": metrics["false_activation_rate"] <= TARGETS["false_activation_rate"],
        "feedback_gain": metrics["feedback_gain"]["ok"],
        "scope_leak": metrics["scope_leak"] == 0,
        "determinism": metrics["determinism"] >= TARGETS["determinism"],
    }
    print(json.dumps({"metrics": metrics, "targets": TARGETS, "meets_targets": targets_ok,
                      "ok": all(targets_ok.values())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
