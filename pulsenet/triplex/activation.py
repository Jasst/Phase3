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
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .journal import Fact, _dt

ACT_STEPS = 3
ACT_TOP_K = 16
ACT_FANOUT = 4
ACT_DECAY = 0.55
ACT_LEARNING = 0.45
ACT_THRESHOLD = 0.002
ACT_REINFORCE = 0.04
ACT_MEMORY = 0.70
BOOST_HALF_LIFE_SEC = 600.0   # 10 мин: подкрепление не должно навсегда закреплять факт в лидерах
BOOST_USAGE_WEIGHT = 0.5      # вклад boost в usage меньше, чем вклад реальных обращений
BOOST_USAGE_CAP = 0.30        # потолок надбавки к весу ребра от usage
# Лишнее звено с активацией ниже доли слабого звена пути — это не гипотеза, а шум.
PATH_SUBSET_ACT_RATIO = 0.5


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
    # Phase4+: путь больше не один. Остальные поля проецируются в ThoughtState.
    paths: List[dict] = field(default_factory=list)
    alternatives: List[dict] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    conflicts: List[dict] = field(default_factory=list)
    beam: int = 1

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
            "paths": self.paths,
            "alternatives": self.alternatives,
            "open_questions": self.open_questions,
            "conflicts": self.conflicts,
        }


def _dot(a, b) -> float:
    try:
        return float(a @ b)
    except (TypeError, ValueError):
        return float(sum(x * y for x, y in zip(a, b)))


def availability(mem, f: Fact, t) -> float:
    """Эпистемический фильтр: можно ли сейчас влиять этим фактом на вывод.

    Помимо confidence учитываются структурированные verdict'ы Phase4+:
    спорный факт, штраф отношения и надёжность источника. Сам журнал при этом
    не меняется — это оценочный слой поверх него.
    """
    if f.retracted or f.invalid_at is not None:
        return 0.0
    if mem.is_archived(f, t):
        return 0.0
    rel = mem.relations.get(f.r, {})
    prior = 0.55 if rel.get("proposed") else 1.0
    prior *= max(0.25, 1.0 - float(rel.get("penalty") or 0.0))
    if f.fact_id in getattr(mem, "_disputed", ()):
        prior *= 0.5
    doc_id = f.source.get("doc_id")
    if doc_id:
        prior *= max(0.3, 1.0 - float(getattr(mem, "_source_penalty", {}).get(doc_id, 0.0)))
    return prior * (0.6 + 0.4 * f.confidence)


def effective_boost(f: Fact, t) -> float:
    """Действующее подкрепление с учётом времени. Журнал хранит исходный boost, здесь он не меняется."""
    b = max(0.0, min(1.0, float(getattr(f, "boost", 0.0) or 0.0)))
    if b <= 0.0:
        return 0.0
    boost_at = getattr(f, "boost_at", None)   # старые записи без времени — как необновлённые
    if not boost_at:
        return b
    try:
        now = t if isinstance(t, datetime) else _dt(t)
        age = max(0.0, (now - _dt(boost_at)).total_seconds())
    except (TypeError, ValueError):
        return b
    return b * (0.5 ** (age / BOOST_HALF_LIFE_SEC))


def edge_weight(mem, f: Fact, t, fact_cos: float = 0.0) -> float:
    """Единый вес синапса: уверенность × время × usage × онтология × кью."""
    w = f.confidence * mem.decay_factor(f, t)
    usage = math.log1p(max(0, f.access_count) + BOOST_USAGE_WEIGHT * effective_boost(f, t))
    w *= 1.0 + min(BOOST_USAGE_CAP, 0.12 * min(usage, 3.0))
    rel = mem.relations.get(f.r, {})
    if not rel.get("proposed"):
        w *= 1.15
    w *= 0.4 + 0.6 * max(0.0, min(1.0, fact_cos))
    return max(0.0, min(1.0, w))


def _relation_vectors(mem, cue_vec) -> Dict[str, float]:
    """Косинус отношения с намерением; несколько подцелей дают максимум, а не среднее.

    ``cue_vec`` — один вектор (старый вызов) или список векторов подцелей.
    Максимум сохраняет направление каждой подцели: размытое среднее превращало
    «понять X и проверить Y» в нечёткий кью, который не менял путь.
    """
    vecs = cue_vec if isinstance(cue_vec, (list, tuple)) else [cue_vec]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return {}
    out: Dict[str, float] = {}
    for name in mem.relations:
        vec = mem.encoder.embed([name])[0]
        out[name] = max(0.0, min(1.0, max(_dot(vec, cv) for cv in vecs)))
    return out


def _reconstruct(parents: Dict[str, str], end: str) -> List[str]:
    """Восстановить путь от seed до конца по карте родителей."""
    path: List[str] = []
    cur, seen = end, set()
    while cur and cur not in seen:
        seen.add(cur)
        path.append(cur)
        cur = parents.get(cur)
    path.reverse()
    return path


def _path_score(ids: List[str], act: Dict[str, float], rmatch: Dict[str, float],
                by_id: Dict[str, Fact]) -> float:
    """Оценка траектории: активация × соответствие отношений цели."""
    if not ids:
        return 0.0
    base = sum(act.get(fid, 0.0) for fid in ids) / len(ids)
    match = sum(rmatch.get(by_id[fid].r, 0.0) for fid in ids if fid in by_id) / len(ids)
    return round(base * (0.6 + 0.4 * match), 6)


def _collapse_weak_supersets(paths: List[dict], act: Dict[str, float]) -> List[dict]:
    """Убрать пути, которые лишь добавляют к другому пути слабое звено.

    ``[a, b]`` и ``[c, a, b]`` — разные по длине, но для интерфейса и обратной
    связи это один и тот же след, если c почти не активирован. Сильно
    активированное звено оставляет путь отдельной гипотезой. Primary никогда
    не удаляется: старый контракт path → UI должен оставаться стабильным.
    """
    by_len = sorted(paths, key=lambda p: (len(p["ids"]), p["ids"]))
    drops: set = set()
    for i, small in enumerate(by_len):
        small_ids = set(small["ids"])
        baseline = [act.get(fid, 0.0) for fid in small_ids]
        if not baseline:
            continue
        floor = PATH_SUBSET_ACT_RATIO * min(baseline)
        for big in by_len[i + 1:]:
            big_ids = set(big["ids"])
            if not small_ids < big_ids or big.get("reason") == "primary":
                continue
            extra = big_ids - small_ids
            if max(act.get(fid, 0.0) for fid in extra) < floor:
                drops.add(id(big))
    return [p for p in paths if id(p) not in drops]


def _alternatives(act: Dict[str, float], parents: Dict[str, str],
                  parent_candidates: Dict[str, List[str]], primary: List[str],
                  rmatch: Dict[str, float], by_id: Dict[str, Fact],
                  beam: int, mem, t) -> Tuple[List[dict], List[dict], List[str]]:
    """Несколько путей вместо одного: разные концы и альтернативные родители.

    ``primary`` (старый путь) всегда остаётся первым — интерфейс и обратная
    связь продолжают работать с тем же контрактом. Остальные траектории
    собираются из вторых best-родителей и соседних по активации концов,
    поэтому они реально существуют в графе, а не выдуманы.
    """
    # Ключ — множество звеньев: [a, b] и [b, a] один путь для оценки и обратной
    # связи. representative остаётся лучшим по score, но primary не перетирается.
    scored: Dict[frozenset, dict] = {}

    def add(ids: List[str], reason: str) -> None:
        if len(ids) < 2:
            return
        key = frozenset(ids)
        cand = {
            "ids": list(ids),
            "score": _path_score(ids, act, rmatch, by_id),
            "reason": reason,
            "activation": round(act.get(ids[-1], 0.0), 4),
        }
        old = scored.get(key)
        if (old is None or reason == "primary"
                or (old["reason"] != "primary"
                    and (-cand["score"], len(cand["ids"]), cand["ids"])
                    < (-old["score"], len(old["ids"]), old["ids"]))):
            scored[key] = cand

    add(primary, "primary")
    ends = sorted(act, key=lambda fid: (-act[fid], fid))[:max(2, min(beam + 1, 8))]
    for end in ends:
        add(_reconstruct(parents, end), "top_end")
    for fid, options in parent_candidates.items():
        if fid not in parents or len(options) < 2:
            continue
        alt = dict(parents)
        alt[fid] = options[1]
        for end in ends[:2]:
            ids = _reconstruct(alt, end)
            add(ids, "alternate_parent")

    ordered = sorted(scored.values(), key=lambda p: (-p["score"], p["ids"]))
    ordered = _collapse_weak_supersets(ordered, act)
    # primary остаётся первым даже если его оценки чуть ниже — это обратная совместимость.
    ordered.sort(key=lambda p: 0 if p["reason"] == "primary" else 1)
    paths = ordered[:max(1, min(beam + 1, 6))]
    alternatives = paths[1:]

    # Открытые вопросы: звенья, которые были кандидатами, но не попали в бюджет.
    questions: List[str] = []
    active = set(act)
    for fid, options in parent_candidates.items():
        for other in options[2:]:
            if other in active or other in questions:
                continue
            f = by_id.get(other)
            if f is not None and len(questions) < 8:
                questions.append(mem._fact_text(f))
    return paths, alternatives, questions


def _conflicts(mem, act: Dict[str, float], by_id: Dict[str, Fact]) -> List[dict]:
    """Активные функциональные факты с разными объектами — сигнал, а не тихий выбор."""
    groups: Dict[Tuple[str, str], List[str]] = {}
    for fid in act:
        f = by_id.get(fid)
        if f is None:
            continue
        rel = mem.relations.get(f.r, {})
        if not (rel.get("functional") or f.functional):
            continue
        groups.setdefault((f.s, f.r), []).append(fid)
    out = []
    for (s, r), ids in sorted(groups.items()):
        if len({by_id[i].o for i in ids}) > 1:
            out.append({"s": s, "r": r, "fact_ids": sorted(ids)})
    return out


def _trim(act: Dict[str, float], top_k: int, threshold: float) -> Dict[str, float]:
    kept = [(fid, a) for fid, a in act.items() if a >= threshold]
    kept.sort(key=lambda kv: (-kv[1], kv[0]))
    return dict(kept[:max(1, top_k)])


def run(mem, seeds: Sequence[str], live: Sequence[Fact], by_id: Dict[str, Fact],
        qv, cue_vec, t, query: str, goal: str = "", steps: int = ACT_STEPS,
        top_k: int = ACT_TOP_K, fanout: int = ACT_FANOUT, beam: int = 1) -> Thought:
    """Прогнать импульс по графу и собрать Thought.

    ``beam`` управляет числом возвращаемых траекторий; основной путь и фронт
    активации остаются прежними, поэтому старые клиенты не ломаются.
    """
    thought = Thought(query=query, goal=goal or "", beam=max(1, min(int(beam or 1), 6)))
    if not seeds:
        return thought

    adj = mem._adjacency(list(live))
    rmatch = _relation_vectors(mem, cue_vec)
    cues = cue_vec if isinstance(cue_vec, (list, tuple)) else [cue_vec]
    cues = [c for c in cues if c is not None] or [qv]

    def cue_cos(fid: str) -> float:
        """Косинус с запросом ИЛИ с любой подцелью: goal меняет вес seed-фактов."""
        return max([max(0.0, mem._fact_cos(fid, qv))]
                   + [max(0.0, mem._fact_cos(fid, c)) for c in cues])

    act: Dict[str, float] = {}
    for fid in seeds:
        f = by_id.get(fid)
        if f is None:
            continue
        cos = cue_cos(fid)
        a = edge_weight(mem, f, t, cos) * availability(mem, f, t)
        if a > 0:
            act[fid] = a
    act = _trim(act, top_k, ACT_THRESHOLD)
    if not act:
        return thought

    parent_candidates: Dict[str, List[Tuple[float, str]]] = {}
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
                    cos = cue_cos(nid)
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
                    if fid != nid:
                        # запасные родители — основа альтернативных путей Phase4+.
                        parent_candidates.setdefault(nid, []).append((contrib, fid))

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
    path = _reconstruct(thought.parents, end)
    thought.path = path

    # Альтернативные траектории и открытые звенья — новая часть Phase4+.
    ranked_parents: Dict[str, List[str]] = {}
    for nid, options in parent_candidates.items():
        seen_p, uniq = set(), []
        for _, parent in sorted(options, key=lambda x: (-x[0], x[1])):
            if parent not in seen_p:
                seen_p.add(parent)
                uniq.append(parent)
        ranked_parents[nid] = uniq[:4]
    thought.paths, thought.alternatives, thought.open_questions = _alternatives(
        act, thought.parents, ranked_parents, path, rmatch, by_id, thought.beam, mem, t)
    thought.conflicts = _conflicts(mem, act, by_id)
    thought.status = "conflict" if thought.conflicts else "ok"
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
    # Один факт = один кредит за мысль, даже если он попал в несколько путей
    # (основной и альтернативный) или передан вызывающим кодом дважды.
    seen: set = set()
    ordered: List[str] = []
    for raw in ordered_fact_ids:
        fid = str(raw)
        if fid in seen:
            continue
        seen.add(fid)
        ordered.append(fid)
    updates: List[Tuple[str, float]] = []
    for i, fid in enumerate(reversed(ordered)):
        amount = ACT_REINFORCE * v * (ACT_MEMORY ** i)
        if abs(amount) < 0.0005:
            break
        if fid in mem.journal.facts:
            updates.append((fid, round(amount, 4)))
    for fid, amount in updates:
        mem.journal.reinforce([fid], amount, at=at)
    return updates
