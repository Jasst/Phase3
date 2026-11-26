# -*- coding: utf-8 -*-
"""Phase4+ · состояние мысли, декомпозиция цели и журнал мыслей.

Слой не заменяет активационный граф из ``activation.py``: он добавляет к нему
четыре вещи, которых не было в первой версии Phase4:

  1. ``split_goal``   — детерминированная декомпозиция цели на подцели;
  2. ``plan_budget``  — адаптивный бюджет импульса под размер графа и сложность цели;
  3. ``ThoughtState`` — многопутевая мысль с рабочим набором, альтернативами,
     открытыми вопросами и структурированным исходом;
  4. ``ThoughtLog``   — append-only журнал мыслей рядом с памятью, чтобы мысль
     переживала перезапуск и была доступна для разбора (`triplex_reflect`).

Журнал мыслей — это не источник истины и не замена журнала фактов: он хранит
только след рассуждения (id фактов, цель, исход). Право на факт по-прежнему
даёт ``doc_id + span`` в ``journal.jsonl``.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

_TOKEN = re.compile(r"\w+", re.UNICODE)
_GOAL_SPLIT = re.compile(r"\s+(?:и|но|а также|чтобы|затем|далее|;)\s+", re.UNICODE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def split_goal(goal: str) -> List[str]:
    """ Разложить цель на подцели без вызова модели.

    ``"чтобы понять X и проверить Y"`` → ``["чтобы понять X", "проверить Y"]``.
    Возвращает ``[goal]``, если разложить нечем: активация должна работать
    и с обычной фразой. Парсер консервативный — лучше оставить одну цель,
    чем нарезать осмысленную фразу на мусор.
    """
    text = (goal or "").strip()
    if not text:
        return []
    parts = [p.strip(" .,;:!?—-") for p in _GOAL_SPLIT.split(text)]
    parts = [p for p in parts if len(_TOKEN.findall(p)) >= 2]
    if len(parts) <= 1:
        return [text]
    # «чтобы понять X» и «проверить Y» — разные кью; документы остаются теми же.
    return parts


def plan_budget(live_count: int, goal: str = "", k_facts: int = 8, steps: int = 3) -> dict:
    """Детерминированный бюджет импульса: сложная цель → длиннее, маленький граф → уже.

    Все значения остаются в границах старого импульса (steps ≤ 8, top_k ≤ 50),
    поэтому старые вызовы без цели ведут себя как раньше.
    """
    subs = split_goal(goal)
    n = max(0, int(live_count))
    out_steps = int(steps)
    if subs and len(subs) > 1:
        out_steps = min(4, max(1, out_steps + 1))
    top_k = max(12, int(k_facts) * 2)
    top_k = min(top_k, max(12, n), 50)
    fanout = 4 if n >= 12 else 3
    return {
        "steps": max(1, min(8, out_steps)),
        "top_k": top_k,
        "fanout": fanout,
        "beam": 3 if subs else 1,
        "subgoals": subs,
    }


@dataclass
class ThoughtState:
    """Расширенный слепок мысли: путь + альтернативы + рабочий набор + исход."""

    thought_id: str = ""
    scope: str = ""
    parent_thought_id: Optional[str] = None
    query: str = ""
    goal: str = ""
    subgoals: List[str] = field(default_factory=list)
    mode: str = "recall"
    status: str = "empty"
    activation: Dict[str, float] = field(default_factory=dict)
    parents: Dict[str, str] = field(default_factory=dict)
    path: List[str] = field(default_factory=list)
    paths: List[dict] = field(default_factory=list)
    alternatives: List[dict] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)
    steps: List[dict] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def from_thought(cls, thought, *, thought_id: str = "", scope: str = "",
                     parent_thought_id: Optional[str] = None,
                     subgoals: Optional[List[str]] = None,
                     mode: str = "recall") -> "ThoughtState":
        """Проекция активационного ``Thought`` в расширенное состояние."""
        paths = list(getattr(thought, "paths", []) or [])
        return cls(
            thought_id=thought_id,
            scope=scope,
            parent_thought_id=parent_thought_id,
            query=getattr(thought, "query", ""),
            goal=getattr(thought, "goal", ""),
            subgoals=list(subgoals or []),
            mode=mode,
            status=getattr(thought, "status", "empty"),
            activation=dict(getattr(thought, "activation", {}) or {}),
            parents=dict(getattr(thought, "parents", {}) or {}),
            path=list(getattr(thought, "path", []) or []),
            paths=paths,
            alternatives=list(getattr(thought, "alternatives", []) or []),
            open_questions=list(getattr(thought, "open_questions", []) or []),
            conflicts=list(getattr(thought, "conflicts", []) or []),
            steps=list(getattr(thought, "steps", []) or []),
        )

    def to_dict(self, mem=None, limit: int = 20) -> dict:
        """Совместимый с UI вид плюс новые поля Phase4+."""
        def line(fid: str) -> dict:
            f = mem.journal.facts.get(fid) if mem is not None else None
            return {
                "fact_id": fid,
                "text": mem._fact_text(f) if f is not None else fid,
                "relation": f.r if f is not None else None,
                "activation": round(self.activation.get(fid, 0.0), 4),
                "source": f.source if f is not None else None,
                "confidence": round(f.confidence, 4) if f is not None else None,
            }

        ranked = sorted(self.activation.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return {
            # legacy-проекция: path/facts/activation/steps остаются на своих местах
            "status": self.status,
            "query": self.query,
            "goal": self.goal,
            "steps": self.steps,
            "path": [line(fid) for fid in self.path],
            "facts": [line(fid) for fid, _ in ranked],
            "activation": {fid: round(a, 4) for fid, a in ranked},
            "path_length": len(self.path),
            # Phase4+
            "thought_id": self.thought_id,
            "scope": self.scope,
            "parent_thought_id": self.parent_thought_id,
            "subgoals": self.subgoals,
            "mode": self.mode,
            "paths": self.paths,
            "alternatives": self.alternatives,
            "open_questions": self.open_questions,
            "conflicts": self.conflicts,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_log_record(self) -> dict:
        return {
            "thought_id": self.thought_id,
            "scope": self.scope,
            "parent_thought_id": self.parent_thought_id,
            "query": self.query,
            "goal": self.goal,
            "subgoals": self.subgoals,
            "mode": self.mode,
            "status": self.status,
            "path_ids": list(self.path),
            "paths": [{"ids": list(p.get("ids") or []), "score": round(float(p.get("score") or 0.0), 4)}
                      for p in self.paths],
            "alternatives": [{"ids": list(p.get("ids") or []), "score": round(float(p.get("score") or 0.0), 4)}
                             for p in self.alternatives][:5],
            "open_questions": self.open_questions[:10],
            "conflicts": self.conflicts[:10],
            "activation_summary": {fid: round(a, 4) for fid, a in
                                   sorted(self.activation.items(), key=lambda kv: (-kv[1], kv[0]))[:10]},
            "outcome": dict(self.outcome),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ThoughtLog:
    """Append-only журнал мыслей: thoughts.jsonl рядом с journal.jsonl.

    Строка ``thought`` — снимок мысли, строка ``event`` — последующее событие
    (например, обратная связь). ``get``/``recent`` склеивают их по thought_id.
    """

    MAX_ROWS = 5000

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._rows: Dict[str, dict] = {}
        self._order: List[str] = []
        self._load()

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue                     # оборванный хвост не ломает журнал
                    self._apply(rec)

    def _apply(self, rec: dict) -> None:
        op = rec.get("op")
        tid = rec.get("thought_id") or ""
        if op == "thought" and tid:
            row = dict(rec)
            row.pop("op", None)
            old = self._rows.get(tid)
            if old:
                row["outcome"] = dict(old.get("outcome") or {})
                row["events"] = list(old.get("events") or [])
            self._rows[tid] = row
            if tid not in self._order:
                self._order.append(tid)
        elif op == "event" and tid in self._rows:
            row = self._rows[tid]
            kind = rec.get("kind") or ""
            events = list(row.get("events") or [])
            events.append({k: v for k, v in rec.items() if k not in ("op", "kind", "thought_id")}
                          | {"kind": kind})
            row["events"] = events[-20:]
            if rec.get("outcome") is not None:
                out = rec["outcome"]
                row["outcome"] = dict(out) if isinstance(out, dict) else {"status": str(out)}
            if rec.get("verdicts") is not None:
                row["verdicts"] = list(rec["verdicts"])
            row["updated_at"] = rec.get("at") or row.get("updated_at")

    def _write(self, rec: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    def append(self, state: "ThoughtState") -> None:
        if not state.thought_id:
            return
        rec = state.to_log_record()
        with self._lock:
            self._write({"op": "thought", **rec})
            self._apply({"op": "thought", **rec})
            self._compact()

    def event(self, thought_id: str, kind: str, **payload) -> None:
        if not thought_id:
            return
        rec = {"op": "event", "thought_id": thought_id, "kind": kind,
               "at": payload.get("at") or _now(), **payload}
        with self._lock:
            self._write(rec)
            self._apply(rec)

    def get(self, thought_id: str) -> Optional[dict]:
        with self._lock:
            row = self._rows.get(thought_id)
            return dict(row) if row else None

    def recent(self, limit: int = 20, goal: str = "", scope: str = "") -> List[dict]:
        with self._lock:
            rows = [self._rows[tid] for tid in reversed(self._order) if tid in self._rows]
        if scope:
            rows = [r for r in rows if not r.get("scope") or r["scope"] == scope]
        if goal:
            needle = goal.lower()
            rows = [r for r in rows if needle in (r.get("goal") or "").lower()
                    or any(needle in g.lower() for g in (r.get("subgoals") or []))]
        return [dict(r) for r in rows[:max(1, min(int(limit), 200))]]

    def _compact(self) -> None:
        """Ограничить рост файла: события старых мыслей сворачиваются в последнюю запись."""
        if len(self._order) <= self.MAX_ROWS:
            return
        keep = set(self._order[-self.MAX_ROWS:])
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for tid in self._order:
                if tid in keep:
                    fh.write(json.dumps({"op": "thought", **self._rows[tid]},
                                        ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, self.path)
        self._rows = {tid: self._rows[tid] for tid in self._order if tid in keep}
        self._order = [tid for tid in self._order if tid in keep]
