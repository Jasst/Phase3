# -*- coding: utf-8 -*-
"""Дедупликация траекторий: permutations, слабые надмножества, кредит фактов."""
from pulsenet.triplex import HashEncoder, TriplexMemory
from pulsenet.triplex.activation import _alternatives, apply_valence


def _paths(act, parents, primary, beam=3, candidates=None):
    """Прямой вызов сборщика путей без запуска всего активационного импульса."""
    return _alternatives(act, parents, candidates or {}, primary, {}, {}, beam, None, None)


def _memory(tmp_path):
    mem = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    doc = mem.docs.add("Маша любит кота. Кот лежит на коврике.")
    src = {"doc_id": doc.doc_id, "span": [0, 5]}
    a = mem.assert_triple("Маша", "любит", "кот", source=src)
    b = mem.assert_triple("кот", "лежит_на", "коврик", source=src)
    return mem, a["fact_id"], b["fact_id"]


def test_reversed_paths_are_one_path():
    act = {"a": 0.5, "b": 0.4}
    paths, alternatives, _ = _paths(act, {"b": "a", "a": "b"}, ["a", "b"])

    assert len(paths) == 1
    assert paths[0]["ids"] == ["a", "b"]      # representative — primary, а не перестановка
    assert paths[0]["reason"] == "primary"
    assert alternatives == []


def test_weak_superset_is_collapsed():
    act = {"a": 0.5, "b": 0.45, "c": 0.05}
    paths, _, _ = _paths(act, {"b": "a", "c": "b"}, ["a", "b"])

    assert [p["ids"] for p in paths] == [["a", "b"]]


def test_strong_superset_stays_a_separate_hypothesis():
    act = {"a": 0.5, "b": 0.45, "c": 0.4}
    paths, alternatives, _ = _paths(act, {"b": "a", "c": "b"}, ["a", "b"])

    assert len(paths) == 2
    assert ["a", "b", "c"] in [p["ids"] for p in paths]
    assert len(alternatives) == 1


def test_primary_stays_first_even_with_lower_score():
    act = {"a": 0.1, "b": 0.1, "c": 0.9, "d": 0.9}
    paths, _, _ = _paths(act, {"b": "a", "d": "c"}, ["a", "b"])

    assert paths[0]["ids"] == ["a", "b"]
    assert paths[0]["reason"] == "primary"
    assert paths[1]["score"] > paths[0]["score"]


def test_best_representative_wins_for_non_primary_duplicate_set():
    # Два разных порядка того же набора: остаётся путь с большей активацией конца.
    act = {"x": 0.1, "y": 0.1, "a": 0.2, "b": 0.9}
    parents = {"y": "x", "b": "a", "a": "b"}
    paths, _, _ = _paths(act, parents, ["x", "y"])

    assert [p["ids"] for p in paths] == [["x", "y"], ["a", "b"]]
    assert paths[1]["reason"] != "primary"


def test_valence_credits_each_fact_once(tmp_path):
    mem, fid_a, fid_b = _memory(tmp_path)
    updates = apply_valence(mem, [fid_a, fid_b, fid_a], 1.0)

    assert [fid for fid, _ in updates] == [fid_b, fid_a]
    assert updates[0][1] > updates[1][1]      # конец пути получает больше
    assert mem.journal.facts[fid_a].boost == updates[1][1]


def test_feedback_verdicts_credit_fact_once(tmp_path):
    mem, fid_a, _ = _memory(tmp_path)
    out = mem.feedback(verdicts=[{"fact_id": fid_a, "valence": 1.0, "reason": "good"},
                                 {"fact_id": fid_a, "valence": 1.0, "reason": "good"}])

    assert out["status"] == "ok"
    assert len(out["updates"]) == 1
    assert mem.journal.facts[fid_a].boost == 0.04
