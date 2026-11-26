# -*- coding: utf-8 -*-
"""Phase4+ (P0+P1): цель в seeds, многопутевая мысль, журнал мыслей, verdict'ы."""
import json

from pulsenet.triplex import HashEncoder, TriplexMemory
from pulsenet.triplex.thoughts import ThoughtLog, plan_budget, split_goal


def _memory(tmp_path):
    mem = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    doc = mem.ingest("Алиса работает в Lab. Lab находится в Москве. "
                     "Алиса пьёт чай. Чай пьют утром.", session_id="s")
    src = {"doc_id": doc["doc_id"], "span": [0, 10]}
    ids = {}
    for s, r, o in [("Алиса", "работает в", "Lab"), ("Lab", "находится в", "Москва"),
                    ("Алиса", "пьёт", "чай"), ("чай", "пьют в", "утро")]:
        res = mem.assert_triple(s, r, o, source=src)
        ids[res["text"]] = res["fact_id"]
    return mem, ids, doc["doc_id"]


def test_goal_decomposition_and_budget_are_deterministic():
    assert split_goal("") == []
    assert split_goal("коврик") == ["коврик"]
    assert split_goal("чтобы понять X и проверить Y") == ["чтобы понять X", "проверить Y"]

    a = plan_budget(40, "чтобы понять X и проверить Y", 8, 3)
    b = plan_budget(40, "чтобы понять X и проверить Y", 8, 3)
    assert a == b
    assert a["steps"] == 4 and a["beam"] == 3 and len(a["subgoals"]) == 2
    small = plan_budget(4, "", 8, 3)
    assert small["fanout"] == 3 and small["beam"] == 1 and small["steps"] == 3
    assert plan_budget(1000, "и", 50, 8)["top_k"] <= 50


def test_goal_changes_which_fact_leads_the_thought(tmp_path):
    mem, ids, _ = _memory(tmp_path)
    work = mem.think("Алиса", goal="где работает Алиса", steps=2, scope="t:work")
    drink = mem.think("Алиса", goal="что пьёт Алиса", steps=2, scope="t:drink")

    assert work["status"] == drink["status"] == "ok"
    assert work["path"][-1]["fact_id"] == ids["Алиса работает в Lab"], work["path"]
    assert drink["path"][-1]["fact_id"] == ids["Алиса пьёт чай"], drink["path"]
    # разные цели дают разные альтернативные продолжения
    assert [p["ids"] for p in work["paths"]] != [p["ids"] for p in drink["paths"]]


def test_thought_has_multiple_paths_and_alternatives(tmp_path):
    mem, _, _ = _memory(tmp_path)
    th = mem.think("Алиса", goal="понять работу и проверить чай", steps=2, scope="t:multi")

    assert len(th["paths"]) >= 2
    assert len(th["alternatives"]) >= 1
    assert [p["fact_id"] for p in th["path"]] == th["paths"][0]["ids"]  # старый контракт не сломан
    unique = {tuple(p["ids"]) for p in th["paths"]}
    assert len(unique) == len(th["paths"])
    row = mem.thoughts.get(th["thought_id"])
    assert row["subgoals"] and row["paths"] and row["thought_id"] == th["thought_id"]


def test_graph_exposes_phase4_paths_without_breaking_legacy(tmp_path):
    mem, ids, _ = _memory(tmp_path)
    th = mem.think("Алиса", goal="понять работу и проверить чай", steps=2, scope="web:ui")
    g = mem.graph(max_edges=50, scope="web:ui")
    t = g["thought"]

    # Legacy-контракт графа: path — список id, как и до Phase4+.
    assert t["thought_id"] == th["thought_id"]
    assert t["path"] == [line["fact_id"] for line in th["path"]]
    assert t["paths"] and t["paths"][0]["ids"] == t["path"]
    assert t["subgoals"] == th["subgoals"]
    assert t["open_fact_ids"] == [] and t["conflicts"] == []

    # Текстовые open_questions проецируются в id только из показанных рёбер.
    stored = mem._thoughts["web:ui"]["thought"]
    stored["open_questions"] = ["Алиса работает в Lab"]
    stored["conflicts"] = [{"s": "Алиса", "r": "работает в",
                            "fact_ids": [ids["Алиса работает в Lab"]]}]
    g2 = mem.graph(max_edges=50, scope="web:ui")["thought"]
    assert g2["open_fact_ids"] == [ids["Алиса работает в Lab"]]
    assert g2["conflicts"][0]["fact_ids"] == [ids["Алиса работает в Lab"]]

    # Старый след без полей Phase4+ тоже читается.
    for key in ("paths", "subgoals", "alternatives", "open_questions", "conflicts"):
        stored.pop(key, None)
    legacy = mem.graph(max_edges=50, scope="web:ui")["thought"]
    assert legacy["path"] == t["path"]
    assert legacy["paths"][0]["ids"] == legacy["path"]
    assert legacy["subgoals"] == [] and legacy["conflicts"] == []
    assert legacy["open_fact_ids"] == [] and legacy["open_questions"] == []


def test_thought_log_survives_restart_and_filters_by_scope_and_goal(tmp_path):
    mem, _, _ = _memory(tmp_path)
    th = mem.think("Алиса", goal="где работает Алиса", scope="mcp:alice")
    tid = th["thought_id"]

    restored = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    assert restored.thoughts.get(tid)["goal"] == "где работает Алиса"
    assert restored.thoughts.recent(5, scope="mcp:bob") == []
    mine = restored.thoughts.recent(5, scope="mcp:alice")
    assert [r["thought_id"] for r in mine] == [tid]
    assert restored.thoughts.recent(5, goal="работает", scope="mcp:alice")
    assert restored.stats()["thoughts_logged"] >= 1
    assert restored.journal.check_invariants() == []


def test_structured_feedback_uses_any_path_and_records_outcome(tmp_path):
    mem, ids, doc_id = _memory(tmp_path)
    th = mem.think("Алиса", goal="понять работу и проверить чай", steps=2, scope="mcp:alice")
    tid = th["thought_id"]
    # берём звено из альтернативного пути — раньше такие id отвергались
    alt_fid = next(fid for p in th["alternatives"] for fid in p["ids"]
                   if fid != th["paths"][0]["ids"][0])
    before = mem.journal.facts[alt_fid].access_count
    mem.feedback([alt_fid], 0.5, thought_id=tid, scope="mcp:alice")
    boosted = mem.journal.facts[alt_fid].boost
    assert boosted > 0

    out = mem.feedback(
        verdicts=[{"fact_id": alt_fid, "valence": -0.5, "reason": "wrong_relation"}],
        overall_valence=-0.5, outcome="partial", thought_id=tid, scope="mcp:alice")

    assert out["status"] == "ok" and out["verdicts"][0]["fact_id"] == alt_fid
    assert mem.journal.facts[alt_fid].access_count == before          # I10
    assert mem.journal.facts[alt_fid].boost < boosted   # журнал не уходит ниже нуля (I10 не тронут)
    assert any(r.get("penalty") for r in mem.relations.values())      # prior relations
    row = mem.thoughts.get(tid)
    assert row["outcome"]["status"] == "partial"
    assert row["events"][-1]["kind"] == "feedback"

    restored = TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))
    assert any(r.get("penalty") for r in restored.relations.values())
    assert restored.journal.check_invariants() == []


def test_verdicts_reject_foreign_fact_and_bad_reason(tmp_path):
    mem, _, _ = _memory(tmp_path)
    th = mem.think("Алиса", goal="коврик", scope="mcp:alice")
    foreign = mem.journal.assert_fact("ёж", "спит", "нора",
                                      source={"doc_id": "d:1", "span": [0, 1]}).fact_id

    bad = mem.feedback(verdicts=[{"fact_id": foreign, "valence": 1, "reason": "good"}],
                       thought_id=th["thought_id"], scope="mcp:alice")
    assert bad["status"] == "rejected"
    assert mem.journal.facts[foreign].boost == 0.0

    unknown = mem.feedback(verdicts=[{"fact_id": th["path"][0]["fact_id"],
                                      "valence": 1, "reason": "магия"}],
                           thought_id=th["thought_id"], scope="mcp:alice")
    assert unknown["status"] == "rejected"


def test_thought_log_compacts_without_losing_last_thought(tmp_path, monkeypatch):
    log = ThoughtLog(str(tmp_path / "thoughts.jsonl"))
    monkeypatch.setattr(ThoughtLog, "MAX_ROWS", 3)
    for i in range(6):
        log.append(type("S", (), {
            "thought_id": f"t{i}", "scope": "", "parent_thought_id": None, "query": "q",
            "goal": "g", "subgoals": [], "mode": "recall", "status": "ok",
            "activation": {}, "parents": {}, "path": [], "paths": [], "alternatives": [],
            "open_questions": [], "conflicts": [], "steps": [], "outcome": {},
            "created_at": "2026-01-01T00:00:00+00:00", "updated_at": "2026-01-01T00:00:00+00:00",
            "to_log_record": lambda self: {
                "thought_id": self.thought_id, "scope": self.scope, "query": self.query,
                "goal": self.goal, "subgoals": [], "paths": [], "path_ids": [],
                "outcome": {}, "created_at": self.created_at, "updated_at": self.updated_at,
            },
        })())
    assert len(log) <= 3
    assert log.get("t5") is not None
    reopened = ThoughtLog(str(tmp_path / "thoughts.jsonl"))
    assert reopened.get("t5") is not None
    assert json.loads((tmp_path / "thoughts.jsonl").read_text(encoding="utf-8").splitlines()[-1])["thought_id"] == "t5"
