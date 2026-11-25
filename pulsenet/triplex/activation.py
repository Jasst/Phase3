# -*- coding: utf-8 -*-
"""Phase4 · Activation-Guided Graph Retrieval (практическая версия).

Это НЕ graph transformer и НЕ backprop в смысле градиентов. Это слой управления,
который запускается на существующем append-only журнале без torch и без обучения
модели:

  * query relevance — что похоже на запрос (уже есть: LSH ∪ Vector → RRF);
  * activation availability — что должен влиять на вывод сейчас;
  * impulse — рекуррентное распространение активации по рёбрам с затуханием,
    веерным ограничением и торможением хабов;
  * thought — активированный подграф + путь с провенансом;
  * valence feedback — обратное кредитование пути (limbic-like), а не
    градиентный спуск.

Формально:

    A[t+1] = B( δ·A[t] + η·Σ W(e)·m(c,e)·A[t] / √fan(v) )

где B — бюджетный проектор (top-k, порог, fan-out), δ — затухание, η — скорость
распространения, W(e) — вес ребра, m(c,e) — соответствие отношения текущему
намерению c, fan(v) — число выходов узла.

Терминология опирается на AGGR (Activation-Guided Graph Retrieval for Cognitive
Memory Reconstruction, arXiv:2611.13127): различие query-conditioned relevance
и activation-conditioned availability, fan-aware traversal, recursive cues и
реконструкция typed evidence graph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .journal import Fact

ACT_STEPS = 3
ACT_TOP_K = 16
ACT_FANOUT = 4
ACT_DECAY = 0.55
ACT_LEARNING = 0.45
ACT_THRESHOLD = 0.002
ACT_REINFORCE = 0.04
ACT_MEMORY = 0.70


@dataclass
class Thought:
    """Слепок мысли: не один факт и не один вектор, а траектория активации."""

    query: str
    goal: str = ""
    status: str = "empty"
    activation: Dict[str, float] = field(default_factory=dict)
    parents: Dict[str, str] = field(default_factory=dict)
    path: List[str] = field(default_factory=list)
    steps: List[dict] = field(default_factory=list)

    def to_dict(self, mem=None, limit: int = 20) -> dict:
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
            "status": self.status,
            "query": self.query,
            "goal": self.goal,
            "steps": self.steps,
            "path": [line(fid) for fid in self.path],
            "facts": [line(fid) for fid, _ in ranked],
            "activation": {fid: round(a, 4) for fid, a in ranked},
            "path_length": len(self.path),
        }


def _dot(a, b) -> float:
    try:
        return float(a @ b)
    except (TypeError, ValueError):
        return float(sum(x * y for x, y in zip(a, b)))


def availability(mem, f: Fact, t) -> float:
    """Эпистемический фильтр: можно ли сейчас влиять этим фактом на вывод."""
    if f.retracted or f.invalid_at is not None:
        return 0.0
    if mem.is_archived(f, t):
        return 0.0
    rel = mem.relations.get(f.r, {})
    prior = 0.55 if rel.get("proposed") else 1.0
    return prior * (0.6 + 0.4 * f.confidence)


def edge_weight(mem, f: Fact, t, fact_cos: float = 0.0) -> float:
    """Единый вес синапса: уверенность × время × usage × онтология × кью."""
    w = f.confidence * mem.decay_factor(f, t)
    usage = math.log1p(max(0, f.access_count) + 2.0 * max(0.0, f.boost))
    w *= 1.0 + 0.12 * min(usage, 3.0)
    rel = mem.relations.get(f.r, {})
    if not rel.get("proposed"):
        w *= 1.15
    w *= 0.4 + 0.6 * max(0.0, min(1.0, fact_cos))
    return max(0.0, min(1.0, w))


def _relation_vectors(mem, cue_vec) -> Dict[str, float]:
    """Косинус отношения с текущим намерением; кэшируется на вызов run()."""
    if cue_vec is None:
        return {}
    out: Dict[str, float] = {}
    for name in mem.relations:
        vec = mem.encoder.embed([name])[0]
        out[name] = max(0.0, min(1.0, _dot(vec, cue_vec)))
    return out


def _trim(act: Dict[str, float], top_k: int, threshold: float) -> Dict[str, float]:
    kept = [(fid, a) for fid, a in act.items() if a >= threshold]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return dict(kept[:max(1, top_k)])


def run(mem, seeds: Sequence[str], live: Sequence[Fact], by_id: Dict[str, Fact],
        qv, cue_vec, t, query: str, goal: str = "", steps: int = ACT_STEPS,
        top_k: int = ACT_TOP_K, fanout: int = ACT_FANOUT) -> Thought:
    """Прогнать импульс по графу и собрать Thought."""
    thought = Thought(query=query, goal=goal or "")
    if not seeds:
        return thought

    adj = mem._adjacency(list(live))
    rmatch = _relation_vectors(mem, cue_vec)
    act: Dict[str, float] = {}
    for fid in seeds:
        f = by_id.get(fid)
        if f is None:
            continue
        cos = max(0.0, mem._fact_cos(fid, qv))
        a = edge_weight(mem, f, t, cos) * availability(mem, f, t)
        if a > 0:
            act[fid] = a
    act = _trim(act, top_k, ACT_THRESHOLD)
    if not act:
        return thought

    for step in range(max(1, min(int(steps), 8))):
        incoming: Dict[str, float] = {}
        parents: Dict[str, Tuple[float, str]] = {}

        for fid, a in list(act.items()):
            if a < ACT_THRESHOLD:
                continue
            f = by_id.get(fid)
            if f is None:
                continue
            for ent in (f.s, f.o):
                edges = adj.get(ent, [])
                if not edges:
                    continue
                scored = []
                for nf in edges:
                    nid = nf.fact_id
                    if nid == fid or nid not in by_id:
                        continue
                    other = nf.o if nf.s == ent else nf.s
                    if other == ent:                     # петлю не распространяем
                        continue
                    cos = max(0.0, mem._fact_cos(nid, qv))
                    w = edge_weight(mem, nf, t, cos) * availability(mem, nf, t)
                    if w <= 0:
                        continue
                    m = 0.5 + 0.5 * rmatch.get(nf.r, 0.0)
                    scored.append((w * m, nid))

                scored.sort(key=lambda x: (-x[0], x[1]))
                for score, nid in scored[:max(1, fanout)]:
                    contrib = a * ACT_LEARNING * score / math.sqrt(len(edges))
                    incoming[nid] = incoming.get(nid, 0.0) + contrib
                    if nid not in parents or contrib > parents[nid][0]:
                        parents[nid] = (contrib, fid)

        nxt: Dict[str, float] = {}
        for fid, a in incoming.items():
            nxt[fid] = ACT_DECAY * act.get(fid, 0.0) + a
        for fid, a in act.items():
            if fid not in nxt:
                nxt[fid] = ACT_DECAY * a
        nxt = _trim(nxt, top_k, ACT_THRESHOLD)

        thought.steps.append({
            "step": step + 1,
            "active": len(nxt),
            "top": [{"fact_id": fid, "activation": round(a, 4)}
                    for fid, a in sorted(nxt.items(), key=lambda kv: (-kv[1], kv[0]))[:5]],
        })
        for nid, (_, parent) in parents.items():
            if parent and parent != nid:
                thought.parents[nid] = parent
        act = nxt
        if not act:
            break

    thought.activation = act
    if not act:
        return thought

    end = sorted(act.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    path: List[str] = []
    cur, seen = end, set()
    while cur and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = thought.parents.get(cur)
    path.reverse()
    thought.path = path
    thought.status = "ok"
    return thought


def apply_valence(mem, ordered_fact_ids: Sequence[str], valence: float,
                  at: Optional[str] = None) -> List[Tuple[str, float]]:
    """Обратное кредитование по пути: результат → причина → seed.

    Положительная валентность усиливает пройденные синапсы, отрицательная —
    ослабляет. Вклад затухает с расстоянием от конца пути. Это не обучение весов
    модели, а оценочный слой поверх журнала; I10 при этом не нарушается.
    """
    v = max(-1.0, min(1.0, float(valence)))
    if not v or not ordered_fact_ids:
        return []
    updates: List[Tuple[str, float]] = []
    for i, fid in enumerate(reversed(list(ordered_fact_ids))):
        amount = ACT_REINFORCE * v * (ACT_MEMORY ** i)
        if abs(amount) < 0.0005:
            break
        if fid in mem.journal.facts:
            updates.append((fid, round(amount, 4)))
    for fid, amount in updates:
        mem.journal.reinforce([fid], amount, at=at)
    return updates
