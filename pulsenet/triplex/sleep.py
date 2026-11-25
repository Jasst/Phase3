# -*- coding: utf-8 -*-
"""«Сон» (M6 брифа): консолидация фактов в веса LoRA.

Этапы (все, кроме train/convert, работают без GPU и проверены тестами):
  1. plan_consolidation(memory)  — отбор eligible-фактов по правилам брифа §8, датасет QA + закрепление.
  2. write_dataset(...)          — датасет в JSONL, фиксированный seed.
  3. accept_adapter(...)         — гейт: старые факты и общая перплексия, иначе откат.
  4. train(...)                  — LoRA r16 α32 lr1e-4 через PEFT. Требует torch, transformers, peft.
  5. convert(...)                — адаптер в GGUF через convert_lora_to_gguf.py из llama.cpp.

Правила eligible (бриф §8): семантическое отношение (τ=∞), conf ≥ 0.8, access_count ≥ 3,
возраст ≥ 7 дней, не вытеснен и не отозван. Эпизодические факты никогда не консолидируются (CLS).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

CONF_MIN = 0.8
ACCESS_MIN = 3
AGE_MIN = timedelta(days=7)
REHEARSAL_RATIO = 0.4          # брифа: 30–50%, берём середину диапазона
SEED = 20260101
LORA = {"r": 16, "alpha": 32, "lr": 1e-4}
GATE_OLD_FACT_DROP = 0.01      # old-fact QA не хуже baseline − 1%
GATE_PPL_RISE = 0.01           # general ppl не выше +1%


class SleepUnavailable(RuntimeError):
    """Обучение недоступно: нет torch/peft/transformers или не задан путь к модели."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def is_eligible(f, memory, now: datetime) -> bool:
    rel = memory.relations.get(f.r)
    if rel is None or rel["tau_days"] is not None:          # только семантические (τ = ∞)
        return False
    if f.retracted or f.invalid_at is not None:              # отозван или вытеснен
        return False
    if f.confidence < CONF_MIN or f.access_count < ACCESS_MIN:
        return False
    return now - _dt(f.valid_from) >= AGE_MIN


LOG_NAME = "sleep_log.jsonl"          # журнал принятых консолидаций (вне журнала фактов)
HOLDOUT_SIZE = 20                     # старые факты, которых нет в обучении: по ним решает гейт


def consolidated_ids(memory) -> set:
    """Факты, уже закреплённые в весах принятым адаптером. Повторно в обучение не берём."""
    path = os.path.join(memory.root, LOG_NAME)
    if not os.path.exists(path):
        return set()
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.update(json.loads(line).get("fact_ids", []))
    return out


def record_consolidation(memory, fact_ids, adapter_dir: str, gate: dict) -> bool:
    """Записать консолидацию, только если гейт принял адаптер. Отклонённый адаптер ничего не меняет."""
    if not gate.get("accept"):
        return False
    rec = {"at": _now().isoformat(), "adapter": adapter_dir, "fact_ids": sorted(fact_ids), "gate": gate}
    with open(os.path.join(memory.root, LOG_NAME), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return True


def plan_consolidation(memory, now: Optional[datetime] = None) -> dict:
    """Собрать датасет для «Сна». Не обучает модель, только отбирает данные.

    Разделение по фактам (без пересечений):
      eligible   — новые семантические факты, уходят в обучение (ответы на них — цель);
      rehearsal  — старые, не закреплённые ранее, повторяются в обучении (40% датасета);
      gate_rows  — отложенная выборка старых фактов, которых НЕТ в обучении: по ней гейт
                   измеряет забывание. Без неё гейт проверял бы запоминание обучающих данных.
    Уже закреплённые факты (sleep_log.jsonl) в обучение не попадают повторно.
    """
    now = now or _now()
    done = consolidated_ids(memory)
    active = [f for f in memory.journal.facts.values() if not f.retracted and f.invalid_at is None]
    eligible = sorted((f for f in active if f.fact_id not in done and is_eligible(f, memory, now)),
                      key=lambda f: f.fact_id)
    eligible_ids = {f.fact_id for f in eligible}
    pool = sorted((f for f in active if f.fact_id not in eligible_ids and f.fact_id not in done),
                  key=lambda f: f.fact_id)
    rng_gate = random.Random(SEED + 1)
    gate_facts = rng_gate.sample(pool, min(HOLDOUT_SIZE, len(pool))) if pool else []
    gate_ids = {f.fact_id for f in gate_facts}
    rehearsal_pool = [f for f in pool if f.fact_id not in gate_ids]
    rng = random.Random(SEED)
    k = min(len(rehearsal_pool), int(len(eligible) * REHEARSAL_RATIO / (1 - REHEARSAL_RATIO))) if eligible else 0
    rehearsal = rng.sample(rehearsal_pool, k) if k else []
    rows = [_qa(memory, f, "new") for f in eligible] + [_qa(memory, f, "rehearsal") for f in rehearsal]
    return {"eligible": len(eligible), "rehearsal": len(rehearsal), "rows": rows,
            "eligible_ids": sorted(eligible_ids),
            "gate_rows": [_qa(memory, f, "gate_old") for f in gate_facts],
            "status": "dataset_ready" if rows else "nothing_eligible"}


def _qa(memory, f, kind: str) -> dict:
    subj = memory._ename(f.s)
    obj = memory._ename(f.o)
    return {"prompt": f"{subj} {f.r}?", "completion": obj, "fact_id": f.fact_id, "kind": kind}


def write_dataset(rows: List[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def accept_adapter(old_before: float, old_after: float, ppl_before: float, ppl_after: float) -> Dict[str, object]:
    """Гейт G7: принять адаптер только если забывание ≤1% и общая перплексия выросла не больше чем на 1%.

    old_before/old_after — точность ответов на gate_rows (доля 0..1), измеренная ДО и ПОСЛЕ обучения.
    ppl_before/ppl_after — перплексия на общей выборке. Измерять должен вызывающий код (см. measure_gate).
    """
    forgetting = old_before - old_after
    ppl_rise = (ppl_after - ppl_before) / ppl_before if ppl_before > 0 else float("inf")
    ok_old = old_after >= old_before - GATE_OLD_FACT_DROP
    ok_ppl = ppl_rise <= GATE_PPL_RISE
    return {"accept": bool(ok_old and ok_ppl), "old_fact_qa": {"before": old_before, "after": old_after},
            "ppl_rise": round(ppl_rise, 4), "old_ok": ok_old, "ppl_ok": ok_ppl}


def train(dataset_path: str, base_dir: str, out_dir: str, steps: int = 200) -> str:
    """LoRA-обучение на HF-версии базовой модели. Запускать на машине с GPU.

    Не тестируется в песочнице: нет torch/peft. Ошибки импорта превращаются в SleepUnavailable.
    """
    if not base_dir or not os.path.isdir(base_dir):
        raise SleepUnavailable("не задан путь к HF-версии базовой модели (base_dir)")
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SleepUnavailable(f"нет зависимостей для обучения: {exc}. Установите torch, transformers, peft") from exc

    try:
        tok = AutoTokenizer.from_pretrained(base_dir)
        model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    except (OSError, ValueError, EnvironmentError) as exc:
        # Новая версия transformers кидает ValueError «Unrecognized model» вместо OSError,
        # если в base_dir нет валидной HF-модели. Для пользователя это тот же «нет стека».
        raise SleepUnavailable(f"не удалось загрузить базовую модель из {base_dir}: {exc}") from exc
    cfg = LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LORA["lr"])
    with open(dataset_path, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    rng = random.Random(SEED)
    model.train()
    for step in range(steps):
        row = rows[rng.randrange(len(rows))]
        prompt_ids = tok(row["prompt"] + " ", return_tensors="pt").input_ids
        full_ids = tok(row["prompt"] + " " + row["completion"], return_tensors="pt").input_ids
        labels = full_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100          # потери только на ответе
        loss = model(input_ids=full_ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    return out_dir


def measure_gate(old_before: float, old_after: float, ppl_before: float, ppl_after: float) -> dict:
    """Единая точка решения: принимает уже измеренные величины и сохраняет их в отчёте гейта."""
    return accept_adapter(old_before, old_after, ppl_before, ppl_after)


def convert(adapter_dir: str, base_dir: str, outfile: str, llama_cpp_dir: str) -> str:
    """Адаптер PEFT → GGUF для llama-server (--lora). Нужен скрипт convert_lora_to_gguf.py из llama.cpp."""
    script = os.path.join(llama_cpp_dir, "convert_lora_to_gguf.py")
    if not os.path.isfile(script):
        raise SleepUnavailable(f"не найден {script}")
    subprocess.run([sys.executable, script, "--base", base_dir, "--outfile", outfile, adapter_dir], check=True)
    return outfile
