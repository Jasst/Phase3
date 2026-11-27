# -*- coding: utf-8 -*-
"""Шум сущностей: фильтр на extract/ingest/assert и чистка старого графа."""
import json

from pulsenet.triplex import HashEncoder, TriplexMemory, parse_extraction
from pulsenet.triplex.memory import is_noise_entity  # реэкспорт, совместимый с патчем cloud
from pulsenet.triplex.noise import noise_reason


def _memory(tmp_path):
    return TriplexMemory(str(tmp_path / "t"), encoder=HashEncoder(256))


def test_pronouns_and_generic_words_are_noise():
    for name in ["ты", "Ты", " я ", "инструменты", "одежду", "вещь",
                 "it", "This", "things", "REQUEST"]:
        assert is_noise_entity(name), name


def test_phrase_fragments_are_noise():
    assert is_noise_entity("на основе моего запроса")
    assert is_noise_entity("с импульсами и итерациями")
    assert is_noise_entity("по результатам проверки")


def test_proper_noun_after_preposition_is_not_noise():
    # Иначе фильтр отсекает нормальные entities вроде «в Dome» или «с Gemma 4».
    for name in ["в Dome", "на Карте", "с Gemma 4", "make-archive.ps1", "Phase3/UI"]:
        assert not is_noise_entity(name), name


def test_real_entities_are_not_noise():
    for name in ["Phase3", "ТРИПЛЕКСИЯ", "память", "Easy Diffusion", "Gemma 4", "Маша"]:
        assert not is_noise_entity(name), name


def test_empty_is_noise_and_reason_codes_are_stable():
    assert noise_reason("  ") == "empty"
    assert noise_reason("Ты") == "generic"
    assert noise_reason("на основе моего запроса") == "fragment"
    assert noise_reason("Phase3") == ""


def test_extra_noise_names_are_configurable():
    assert not is_noise_entity("посредник")
    assert is_noise_entity("Посредник", extra_names={"посредник"})
    assert not is_noise_entity("посредник")


def test_parse_extraction_drops_noise_before_ingest():
    text = "я сделал это. Маша любит кота."
    raw = json.dumps({"entities": [{"name": "ты", "type": "other"},
                                   {"name": "Маша", "type": "person"}],
                      "facts": [{"s": "я", "r": "сделал", "o": "это",
                                 "evidence": "я сделал это", "conf": 0.9},
                                {"s": "Маша", "r": "любит", "o": "кот",
                                 "evidence": "Маша любит кота", "conf": 0.9}]})
    out = parse_extraction(raw, text)

    assert [e["name"] for e in out["entities"]] == ["Маша"]
    assert len(out["facts"]) == 1
    assert out["facts"][0]["s"] == "Маша"


def test_assert_triple_rejects_noise_and_does_not_create_entity(tmp_path):
    mem = _memory(tmp_path)
    doc = mem.docs.add("ты любит кот. Маша любит кота.")
    src = {"doc_id": doc.doc_id, "span": [0, 5]}

    bad = mem.assert_triple("ты", "любит", "кот", source=src)
    assert bad["status"] == "rejected_noise"
    assert bad["reason"] == "generic"
    assert all(e["canonical"] != "ты" for e in mem.entities.values())

    good = mem.assert_triple("Маша", "любит", "кот", source=src)
    assert good["status"] == "ok"


def test_ingest_marks_noise_facts_and_keeps_real_ones(tmp_path):
    mem = _memory(tmp_path)
    text = "я сделал это. Маша любит кота."

    def extractor(_):
        return {"entities": [{"name": "я", "type": "other"},
                             {"name": "Маша", "type": "person"}],
                "facts": [{"s": "я", "r": "сделал", "o": "это",
                           "evidence": "я сделал это", "conf": 0.9},
                          {"s": "на основе моего запроса", "r": "понятно", "o": "это",
                           "evidence": "я сделал это", "conf": 0.9},
                          {"s": "Маша", "r": "любит", "o": "кот",
                           "evidence": "Маша любит кота", "conf": 0.9}]}

    out = mem.ingest(text, extractor=extractor)
    statuses = [f["status"] for f in out["facts"]]
    assert statuses.count("rejected_noise") == 2
    assert statuses.count("ok") == 1
    assert all(e["canonical"] not in ("я", "на основе моего запроса")
               for e in mem.entities.values())


def test_clean_noise_is_dry_by_default_and_retracts_when_asked(tmp_path):
    mem = _memory(tmp_path)
    doc = mem.docs.add("ты любит кот. Маша любит кота.")
    src = {"doc_id": doc.doc_id, "span": [0, 5]}
    noisy = mem._create_entity("ты", "other")
    real = mem._create_entity("Маша", "other")
    mem.journal.assert_fact(noisy, "любит", real, source=src)
    mem.journal.assert_fact(real, "любит", noisy, source=src)

    preview = mem.clean_noise()
    assert preview["dry_run"] is True
    assert len(preview["retracted"]) == 2
    assert preview["entities_marked"][0]["name"] == "ты"
    assert len(mem.journal.query()) == 2              # журнал не тронут

    applied = mem.clean_noise(dry_run=False)
    assert len(applied["retracted"]) == 2
    assert len(mem.journal.query()) == 0
    assert mem.stats()["entities_noisy"] == 1
    assert mem.journal.check_invariants() == []

    # Помеченный узел больше не связывается с новыми фактами.
    action, eid, _, _ = mem.link_entity("ты")
    assert action == "create" and eid is None
