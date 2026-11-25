# -*- coding: utf-8 -*-
"""Phase4: активационный поток, мысль и обратное кредитование пути."""
from datetime import timedelta

import pytest

from pulsenet.triplex import HashEncoder, TriplexMemory
from pulsenet.triplex.activation import BOOST_HALF_LIFE_SEC, effective_boost
from pulsenet.triplex.journal import FactJournal, _dt
from pulsenet.triplex.memory import THOUGHT_HALF_LIFE_SEC


def _memory(tmp_path):
    mem = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    doc = mem.ingest("Маша любит кота. Кот лежит на коврике.", session_id="s")
    src = {"doc_id": doc["doc_id"], "span": [0, 1]}
    a = mem.assert_triple("Маша", "любит", "кот", source=src)
    b = mem.assert_triple("кот", "лежит_на", "коврик", source=src)
    assert a["status"] == "ok" and b["status"] == "ok"
    return mem, a["fact_id"], b["fact_id"]


def test_impulse_reaches_two_hop_fact_and_builds_thought(tmp_path):
    mem, fid_a, fid_b = _memory(tmp_path)

    thought = mem.think("Маша", goal="что лежит у кота", steps=2)

    assert thought["status"] == "ok"
    assert fid_b in thought["activation"], thought
    assert len(thought["path"]) >= 2
    assert any("лежит_на" in f["text"] for f in thought["facts"])
    assert thought["steps"] and thought["steps"][0]["active"] >= 1


def test_old_recall_path_stays_inert_without_activation(tmp_path):
    mem, _, _ = _memory(tmp_path)
    r = mem.recall("Маша")
    assert r["thought"] is None
    assert r["facts"]


def test_feedback_is_backward_and_does_not_touch_access_counter(tmp_path):
    mem, fid_a, fid_b = _memory(tmp_path)
    before = mem.journal.facts[fid_a].access_count

    out = mem.feedback([fid_a, fid_b], 0.7)

    assert out["updates"]
    assert mem.journal.facts[fid_b].boost > mem.journal.facts[fid_a].boost  # конец пути сильнее
    assert mem.journal.facts[fid_a].access_count == before                  # I10 не нарушен
    assert mem.journal.check_invariants() == []


def test_negative_feedback_clamps_boost_to_zero(tmp_path):
    mem, fid_a, fid_b = _memory(tmp_path)
    mem.feedback([fid_a, fid_b], 1.0)
    mem.feedback([fid_a, fid_b], -1.0)
    assert mem.journal.facts[fid_a].boost == 0.0
    assert mem.journal.facts[fid_b].boost == 0.0


def test_boost_survives_replay(tmp_path):
    mem, fid_a, _ = _memory(tmp_path)
    mem.feedback([fid_a], 0.5)
    boost = mem.journal.facts[fid_a].boost

    restored = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    assert restored.journal.facts[fid_a].boost == boost


def test_graph_exposes_weight_boost_and_last_impulse(tmp_path):
    mem, _, _ = _memory(tmp_path)
    mem.think("Маша", goal="коврик", steps=2)

    g = mem.graph(50)
    edge = next(e for e in g["edges"] if e["label"] == "любит")
    assert 0.0 <= edge["weight"] <= 1.0
    assert "boost" in edge
    assert g["thought"]["status"] == "ok"
    assert g["thought"]["path"]                      # путь отдан интерфейсу для подсветки
    assert any(e["activation"] > 0 for e in g["edges"])


def test_thought_activation_decays_and_then_clears(tmp_path, monkeypatch):
    mem, _, _ = _memory(tmp_path)
    clock = [1000.0]
    monkeypatch.setattr("pulsenet.triplex.memory.time.monotonic", lambda: clock[0])
    mem.think("Маша", goal="коврик", steps=2)

    def peak():
        return max([e["activation"] for e in mem.graph(50)["edges"]] or [0.0])

    a0 = peak()
    assert a0 > 0

    clock[0] += THOUGHT_HALF_LIFE_SEC
    a1 = peak()
    assert 0 < a1 < a0
    assert a1 == pytest.approx(a0 / 2, rel=0.15)

    clock[0] += 120
    g = mem.graph(50)
    assert g["thought"] is None
    assert all(e["activation"] == 0 for e in g["edges"])


def test_adjacency_cache_respects_as_of(tmp_path):
    """Кэш смежности, построенный для прошлого, не должен ломать поиск в настоящем."""
    mem = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    src = {"doc_id": "d:1", "span": [0, 1]}
    mem.journal.assert_fact("e1", "любит", "e2", source=src,
                            valid_from="2026-01-01T00:00:00+00:00")
    stale = mem.journal.assert_fact("e2", "лежит", "e3", source=src,
                                    valid_from="2026-01-01T00:00:00+00:00",
                                    valid_to="2026-02-01T00:00:00+00:00")
    reopened = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))

    jan = reopened.recall("e1", as_of="2026-01-15T00:00:00+00:00", activate=False, depth=2)
    assert stale.fact_id in [f["fact_id"] for f in jan["facts"]]

    # Старый код здесь падал: adj из кэша содержал факт, которого нет в by_id.
    mar = reopened.recall("e1", as_of="2026-03-01T00:00:00+00:00", activate=False, depth=2)
    assert stale.fact_id not in [f["fact_id"] for f in mar["facts"]]


def test_feedback_accepts_only_facts_from_thought(tmp_path):
    mem, fid_a, fid_b = _memory(tmp_path)
    thought = mem.think("Маша", goal="коврик", steps=2, scope="mcp:tester")
    tid = thought["thought_id"]
    foreign = mem.journal.assert_fact("ёж", "спит", "нора",
                                      source={"doc_id": "d:1", "span": [0, 1]}).fact_id

    bad = mem.feedback([foreign], 1.0, scope="mcp:tester")
    assert bad["status"] == "rejected"
    assert mem.journal.facts[foreign].boost == 0.0

    good = mem.feedback([fid_b], 0.5, scope="mcp:tester")
    assert good["status"] == "ok"
    assert mem.journal.facts[fid_b].boost > 0

    # Явно указанный id мысли действует и из другого scope, неизвестный — нет.
    assert mem.feedback([fid_a], 0.3, thought_id=tid, scope="mcp:other")["status"] == "ok"
    assert mem.feedback([fid_a], 0.3, thought_id="нет-такой", scope="mcp:other")["status"] == "rejected"


def test_boost_decays_in_weight_but_not_in_journal(tmp_path):
    mem, fid_a, _ = _memory(tmp_path)
    mem.feedback([fid_a], 1.0)
    f = mem.journal.facts[fid_a]
    b0 = f.boost
    t0 = _dt(f.boost_at)

    assert effective_boost(f, t0) == pytest.approx(b0)
    later = t0 + timedelta(minutes=BOOST_HALF_LIFE_SEC / 60)
    assert effective_boost(f, later) == pytest.approx(b0 / 2)
    assert f.boost == b0                         # журнал остался неизменным


def test_thought_traces_are_isolated_by_scope(tmp_path):
    mem, _, _ = _memory(tmp_path)
    mem.think("Маша", goal="коврик", steps=2, scope="alice")
    assert mem.graph(scope="alice")["thought"]["status"] == "ok"
    assert mem.graph(scope="bob")["thought"] is None

    mem.think("Маша", goal="коврик", steps=2, scope="bob")
    assert mem.graph(scope="alice")["thought"]["status"] == "ok"
    assert mem.graph(scope="bob")["thought"]["status"] == "ok"


def test_journal_rejects_boost_out_of_range():
    j = FactJournal()
    f = j.assert_fact("кот", "спит", "диван", source={"doc_id": "d:1", "span": [0, 1]},
                      valid_from="2026-01-01T00:00:00+00:00")
    j.reinforce([f.fact_id], 10.0)
    assert f.boost == 1.0
    j.reinforce([f.fact_id], -10.0)
    assert f.boost == 0.0
    assert j.check_invariants() == []
