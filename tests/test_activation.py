# -*- coding: utf-8 -*-
"""Phase4: активационный поток, мысль и обратное кредитование пути."""
from pulsenet.triplex import HashEncoder, TriplexMemory
from pulsenet.triplex.journal import FactJournal


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


def test_journal_rejects_boost_out_of_range():
    j = FactJournal()
    f = j.assert_fact("кот", "спит", "диван", source={"doc_id": "d:1", "span": [0, 1]},
                      valid_from="2026-01-01T00:00:00+00:00")
    j.reinforce([f.fact_id], 10.0)
    assert f.boost == 1.0
    j.reinforce([f.fact_id], -10.0)
    assert f.boost == 0.0
    assert j.check_invariants() == []
