# -*- coding: utf-8 -*-
"""Регрессионные тесты исправлений: прямая запись находится поиском, дубли не плодятся,
индекс растёт амортизированно, кэши согласованы с журналом, гейт «Сна» ведёт учёт."""
import json
import os
import time

import numpy as np
import pytest

from pulsenet.triplex import HashEncoder, TriplexMemory, make_extractor
from pulsenet.triplex import sleep
from pulsenet.triplex.index import VectorIndex
from pulsenet.triplex.journal import FactJournal

T0 = "2026-01-01T00:00:00+00:00"
T1 = "2026-02-01T00:00:00+00:00"
TEXT = "Нел живёт в Амстердаме. Кот Барсик спит на диване."


def fake_llm(prompt):
    return json.dumps({
        "entities": [{"name": "Нел", "type": "person"}, {"name": "Барсик", "type": "animal"}],
        "facts": [
            {"s": "Нел", "r": "живёт_в", "o": "Амстердаме",
             "evidence": "Нел живёт в Амстердаме.", "conf": 0.9},
            {"s": "Барсик", "r": "спит_на", "o": "диване",
             "evidence": "Кот Барсик спит на диване.", "conf": 0.9},
        ]}, ensure_ascii=False)


@pytest.fixture()
def mem(tmp_path):
    return TriplexMemory(str(tmp_path / "triplex"), encoder=HashEncoder(256),
                         extractor=make_extractor(fake_llm))


def _source(mem, text_fragment="Кот Барсик спит на диване"):
    doc_id = mem.ingest(TEXT, session_id="s")["doc_id"]
    txt = mem.docs.docs[doc_id].text
    a = txt.find(text_fragment)
    return {"doc_id": doc_id, "span": [a, a + len(text_fragment)]}


# ---------- 1. прямая запись (MCP triplex_assert) ----------

def test_direct_assert_is_findable_without_restart(mem):
    src = _source(mem)
    out = mem.assert_triple("Барсик", "любит", "рыбу", source=src)
    assert out["status"] == "ok"
    res = mem.recall("Что любит Барсик?")
    texts = [f["text"] for f in res["facts"]]
    assert any("рыбу" in t for t in texts), texts


def test_direct_assert_links_existing_entity(mem):
    """Субъект должен быть тем же узлом, что и при ingest, а не сырой строкой."""
    mem.ingest(TEXT, session_id="s")
    src = _source(mem)
    out = mem.assert_triple("Барсик", "любит", "рыбу", source=src)
    fact = mem.journal.facts[out["fact_id"]]
    assert fact.s.startswith("e:"), fact.s
    ingest_fact = next(f for f in mem.journal.facts.values() if f.r == "спит_на")
    assert fact.s == ingest_fact.s


def test_direct_assert_normalizes_relation(mem):
    src = _source(mem)
    mem.define_relation("любит", aliases=["нравится"])
    out = mem.assert_triple("Барсик", "нравится", "рыбу", source=src)
    assert out["relation"] == "любит"


# ---------- 2. дедупликация ----------

def test_same_triple_from_two_documents_is_one_fact(mem):
    mem.ingest("Нел живёт в Амстердаме.", session_id="a")
    mem.ingest("Нел живёт в Амстердаме.", session_id="b")
    live = [f for f in mem.journal.query() if f.r == "живёт_в"]
    assert len(live) == 1, [f.fact_id for f in live]
    assert len(live[0].corroborations) == 1       # второй источник сохранён как подтверждение


def test_repeated_ingest_raises_confidence_capped(mem):
    for i in range(6):
        mem.ingest("Нел живёт в Амстердаме.", session_id=f"s{i}")
    live = [f for f in mem.journal.query() if f.r == "живёт_в"]
    assert len(live) == 1
    assert live[0].confidence <= 0.95
    assert live[0].confidence > 0.72               # подтверждения подняли уверенность


def test_corroboration_survives_restart_and_replay_is_identical(tmp_path):
    root = str(tmp_path / "t")
    m = TriplexMemory(root, encoder=HashEncoder(256), extractor=make_extractor(fake_llm))
    m.ingest("Нел живёт в Амстердаме.", session_id="a")
    m.ingest("Нел живёт в Амстердаме.", session_id="b")
    h1 = m.journal.state_hash()
    m2 = TriplexMemory(root, encoder=HashEncoder(256))
    assert m2.journal.state_hash() == h1
    f = next(f for f in m2.journal.facts.values() if f.r == "живёт_в")
    assert len(f.corroborations) == 1


def test_direct_assert_repeat_does_not_duplicate(mem):
    src = _source(mem)
    a = mem.assert_triple("Барсик", "любит", "рыбу", source=src)
    b = mem.assert_triple("Барсик", "любит", "рыбу", source={"doc_id": src["doc_id"], "span": src["span"]})
    assert a["fact_id"] == b["fact_id"]
    assert len([f for f in mem.journal.query() if f.r == "любит"]) == 1


def test_different_objects_are_still_separate(mem):
    src = _source(mem)
    mem.assert_triple("Барсик", "любит", "рыбу", source=src)
    mem.assert_triple("Барсик", "любит", "молоко", source=src)
    assert len([f for f in mem.journal.query() if f.r == "любит"]) == 2


def test_functional_relation_still_supersedes_not_dedupes(tmp_path):
    j = FactJournal()
    src1 = {"doc_id": "d:1", "span": [0, 4]}
    src2 = {"doc_id": "d:2", "span": [0, 4]}
    j.assert_fact("кот", "живёт_в", "доме", source=src1, valid_from=T0, functional=True)
    f2 = j.assert_fact("кот", "живёт_в", "лесу", source=src2, valid_from=T1, functional=True)
    assert f2.supersedes is not None
    assert j.check_invariants() == []


def test_invalid_corroboration_source_is_reported_by_invariants():
    j = FactJournal()
    f = j.assert_fact("кот", "видит", "мышь", source={"doc_id": "d:1", "span": [0, 1]}, valid_from=T0)
    j.corroborate(f.fact_id, source={"doc_id": "d:2", "span": [0, 1]})
    assert j.check_invariants() == []
    f.corroborations.append({"span": [0, 1]})      # повреждение: нет doc_id
    assert any("подтверждение без источника" in v for v in j.check_invariants())


# ---------- 3. индекс ----------

def test_vector_index_growth_keeps_topk_correct():
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((300, 32)).astype(np.float32)
    vi = VectorIndex(32)
    for i, v in enumerate(vecs):
        vi.add(i, v)                               # несколько удвоений буфера
    q = vecs[123]
    top = vi.topk(q, 1)
    assert top[0][0] == 123
    assert vi.mat.shape == (300, 32)
    vi.add(5, vecs[0])                             # переопределение ключа не растит индекс
    assert len(vi) == 300


def test_vector_index_insert_is_fast_enough():
    vi = VectorIndex(512)
    rng = np.random.default_rng(0)
    t = time.time()
    for i in range(8000):
        vi.add(i, rng.standard_normal(512).astype(np.float32))
    assert time.time() - t < 3.0                   # было ~11 с на vstack


# ---------- 4. кэши чтения ----------

def test_chunk_cache_sees_new_documents(mem):
    mem.ingest("Совсем новый документ про облака.", session_id="z")
    assert any("облака" in c.text for c in mem._chunk_by_id.values())


def test_adjacency_cache_refreshes_after_write(mem):
    src = _source(mem)
    mem.recall("Барсик")                            # прогреть кэш смежности
    mem.assert_triple("Барсик", "любит", "рыбу", source=src)
    res = mem.recall("Барсик рыба")
    assert any("рыбу" in f["text"] for f in res["facts"])


def test_access_writes_do_not_fsync_and_still_count(mem):
    src = _source(mem)
    mem.assert_triple("Барсик", "любит", "рыбу", source=src)
    mem.recall("Что любит Барсик?")
    fid = next(f.fact_id for f in mem.journal.facts.values() if f.r == "любит")
    assert mem.journal.facts[fid].access_count == 1


# ---------- 5. «Сон»: учёт и отложенный гейт ----------

def _many_facts(mem, n):
    """Семантическое отношение (τ=∞), факты давние и часто использованы — подходят «Сну»."""
    mem.define_relation("связан_с")
    for i in range(n):
        out = mem.assert_triple(f"Объект{i}", "связан_с", f"Цель{i}",
                                source={"doc_id": "d:fix", "span": [0, 1]})
        f = mem.journal.facts[out["fact_id"]]
        f.access_count = 5
        f.valid_from = "2025-01-01T00:00:00+00:00"


def test_gate_rows_are_disjoint_from_training_rows(tmp_path):
    mem = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    _many_facts(mem, 60)
    plan = sleep.plan_consolidation(mem, now=sleep._now() + sleep.AGE_MIN * 2)
    train_ids = {r["fact_id"] for r in plan["rows"]}
    gate_ids = {r["fact_id"] for r in plan["gate_rows"]}
    assert gate_ids, "отложенная выборка должна существовать"
    assert train_ids.isdisjoint(gate_ids)


def test_consolidated_facts_are_not_planned_again(tmp_path):
    mem = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    _many_facts(mem, 30)
    later = sleep._now() + sleep.AGE_MIN * 2
    first = sleep.plan_consolidation(mem, now=later)
    assert first["eligible_ids"]
    ok = sleep.record_consolidation(mem, first["eligible_ids"], "adapter/run1",
                                    {"accept": True})
    assert ok
    second = sleep.plan_consolidation(mem, now=later)
    assert set(second["eligible_ids"]).isdisjoint(first["eligible_ids"])


def test_rejected_adapter_records_nothing(tmp_path):
    mem = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    _many_facts(mem, 30)
    later = sleep._now() + sleep.AGE_MIN * 2
    plan = sleep.plan_consolidation(mem, now=later)
    gate = sleep.measure_gate(0.9, 0.8, 10.0, 10.0)     # забыл 10%: отказ
    assert gate["accept"] is False
    assert sleep.record_consolidation(mem, plan["eligible_ids"], "adapter/bad", gate) is False
    assert sleep.consolidated_ids(mem) == set()
