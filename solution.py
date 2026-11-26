"""solution.py — точка входа агента на платформе.

Запуск (интерфейс как в baseline.py):
    OPENAI_API_KEY=... OPENAI_MODEL=... OPENAI_BASE_URL=... python solution.py path/to/task.md
Результат: submission.csv рядом с task.md. Рабочие файлы — в agent_work/ рядом с task.md.

Пайплайн:
    1. Исследование задачи      task_understanding  → task_config.json
    2. Первичный анализ данных  eda + data_prep     → eda_profile.json, eda_summary.md, cache/data.pkl
       Схема валидации           validation_plan     → folds.npz (фиксирована на весь запуск)
       Fallback-сабмит (константа) — в первые секунды, чтобы файл существовал всегда
    3–5. Research loop: гипотеза (hypothesis_engine) → эксперимент (ml_experiment) →
         анализ ошибок + рефлексия (ml_analysis) → новый лучший? → перезаписать submission.csv
    6. Контроль ресурсов (resource_controller): время, токены, таймауты; финальный резерв на
       бленд, постобработку, refit и финальный сабмит.
Любая ошибка оркестратора не оставляет задачу без ответа: сабмит уже записан на шаге 2,
а при аварии до этого копируется sample_submission.
"""

from __future__ import annotations

import time

T0 = time.time()

import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent_utils import fmt_duration, fmt_num, log, md_table, set_log_file, short, write_json, write_text  # noqa: E402
from data_prep import prepare_data  # noqa: E402
from eda import profile_dataset, summarize_for_llm  # noqa: E402
from hypothesis_engine import HypothesisEngine  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from ml_analysis import analyze_experiment_errors, history_rows, write_reflection  # noqa: E402
from ml_experiment import (ExperimentStore, Workspace, generate_submission, refit_on_full, run_tuning,  # noqa: E402
                           run_ml_experiment, write_fallback_submission)
from researcher_bridge import ResearcherBridge  # noqa: E402
from resource_controller import ResourceController  # noqa: E402
from task_metrics import default_metric, resolve_metric  # noqa: E402
from task_understanding import load_task_data, understand_task  # noqa: E402
from validation_plan import build_validation  # noqa: E402


def after_experiment(rec: dict, ws: Workspace, store: ExperimentStore, engine: HypothesisEngine, llm, X) -> None:
    """Анализ ошибок → рефлексия → сабмит, если результат новый лучший."""
    try:
        analysis = analyze_experiment_errors(rec["exp_id"], ws, store, X=X) if rec["status"] == "success" else \
            {"report_md": f"ошибка: {short(rec.get('error_message'), 800)}", "facts": {}}
        if rec["status"] == "success":
            engine.last_analysis = analysis["report_md"]
        reflection = write_reflection(rec, analysis, ws, store, llm)
        engine.last_reflection = reflection
        log(f"Рефлексия {rec['exp_id']}: {short(reflection, 300)}")
    except Exception as e:  # noqa: BLE001
        log(f"Анализ {rec['exp_id']} упал: {type(e).__name__}: {short(e, 300)}")
    if rec["status"] == "success" and rec.get("is_new_best"):
        generate_submission(ws, store)


def ensure_refit(rec: dict, ws: Workspace, store: ExperimentStore) -> bool:
    """Получить test-предсказания эксперимента, обученного на всём train (рекурсивно для бленда/постобработки)."""
    if rec is None:
        return False
    if rec.get("test_refit_path"):
        return True
    kind = rec.get("kind")
    ok = False
    if kind in ("registry", "code"):
        ok = refit_on_full(rec, ws)
    elif kind == "blend":
        tests = []
        for cid, w in zip(rec.get("components", []), rec.get("weights", [])):
            if w < 1e-6:  # компонент с нулевым весом не переобучаем
                tests.append(0.0)
                continue
            comp = store.get(cid)
            if not ensure_refit(comp, ws, store):
                log("refit бленда: не все компоненты переобучены — оставляю фолдовые предсказания")
                return False
            tests.append(w * np.load(comp["test_refit_path"]))
        path = Path(rec["test_pred_path"]).parent / "test_refit.npy"
        np.save(path, sum(tests))
        rec["test_refit_path"] = str(path)
        ok = True
    elif kind == "postprocess":
        base = store.get(rec.get("base_exp_id"))
        if ensure_refit(base, ws, store):
            t = np.load(base["test_refit_path"]) * np.asarray(rec["class_weights"])
            t /= t.sum(axis=1, keepdims=True)
            path = Path(rec["test_pred_path"]).parent / "test_refit.npy"
            np.save(path, t)
            rec["test_refit_path"] = str(path)
            ok = True
    store.save()
    return ok


def refit_best(ws: Workspace, store: ExperimentStore) -> None:
    """Для временной валидации фолды не видели последний период — переобучаем лучшее решение на всём train."""
    best = store.best()
    if best is not None:
        ensure_refit(best, ws, store)


def final_stage(ws: Workspace, store: ExperimentStore, engine: HypothesisEngine, llm, X) -> None:
    log(f"Финальная стадия: {ws.rc.status_text()}")
    for _ in range(2):  # бленд и постобработка, если ещё не сделаны для текущего набора
        if ws.rc.remaining() < 60:
            break
        h = engine._automatic()
        if h is None:
            break
        rec = run_ml_experiment(h, ws, store, source="auto", final_stage=True)
        after_experiment(rec, ws, store, engine, None, X)
    if ws.plan.scheme == "time_forward":
        refit_best(ws, store)  # refit_on_full сам проверяет, хватает ли времени
    res = generate_submission(ws, store)
    if res.get("status") != "success":
        # лучший не прошёл проверку формата — берём следующий по качеству
        ranked = sorted(store.successes(), key=lambda h: h["local_cv_score"], reverse=ws.metric.greater_is_better)
        for h in ranked[1:]:
            if generate_submission(ws, store, h["exp_id"]).get("status") == "success":
                break


def write_summary(ws: Workspace, store: ExperimentStore, eda_summary: str) -> None:
    best = store.best()
    rows = history_rows(store.history, last_n=1000)
    text = "\n".join([
        f"# Итог запуска: {Path(ws.cfg.task_md).parent.name}",
        "", f"- задача: {ws.cfg.task_type}, таргет `{ws.cfg.target}`, метрика **{ws.metric.name}** ({ws.metric.direction})",
        f"- формат ответа: {ws.cfg.prediction_format}, колонки {ws.cfg.submission_columns}",
        f"- валидация: {ws.plan.description}",
        f"- лучший: {best['exp_id'] if best else '—'} = {fmt_num(best['local_cv_score']) if best else '—'} "
        f"({short(best['hypothesis_summary'], 200) if best else ''})",
        f"- ресурсы: {ws.rc.status_text()}",
        "", "## EDA", eda_summary,
        "", "## Эксперименты", md_table(rows, ["exp_id", "kind", "summary", "status", "score", "delta", "sec", "reflection"]),
    ])
    write_text(ws.work_dir / "run_summary.md", text)
    write_json(ws.work_dir / "resources.json", ws.rc.to_dict())


def run(task_path: str) -> None:
    rc = ResourceController(start_time=T0)
    task_md = Path(task_path).resolve()
    work = task_md.parent / "agent_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    (work / "cache").mkdir(parents=True, exist_ok=True)
    set_log_file(work / "agent.log")
    log(f"Старт агента: {task_md}; бюджет {fmt_duration(rc.budget_sec)}, CPU {rc.n_threads}")
    llm = LLMClient(rc, work / "llm_log.jsonl")

    # 1. исследование задачи
    data = load_task_data(task_md)
    cfg = understand_task(task_md, llm, work, data)
    _, train, test, sample = data
    metric = resolve_metric(cfg.metric, cfg.task_type) or default_metric(cfg.task_type, cfg.prediction_format)

    # 2. EDA, подготовка данных, валидация
    profile = profile_dataset(cfg, train, test)
    write_json(work / "eda_profile.json", profile)
    eda_summary = summarize_for_llm(profile, cfg)
    write_text(work / "eda_summary.md", eda_summary)
    prepared = prepare_data(cfg, train, test, profile)
    pd.to_pickle(prepared, work / "cache" / "data.pkl")
    plan = build_validation(cfg, prepared["X"], prepared["y"], profile)
    plan.save(work)
    ws = Workspace(cfg=cfg, work_dir=work, meta=prepared["meta"], plan=plan, rc=rc, y=prepared["y"],
                   test_ids=prepared["test_ids"], sample=sample, metric=metric)
    write_fallback_submission(ws)
    researcher = ResearcherBridge(cfg, metric, rc, work, train, test)
    X = prepared["X"]
    del train, test, data

    # 3–5. research loop
    store = ExperimentStore(work, metric)
    engine = HypothesisEngine(ws, store, llm, profile, eda_summary, researcher)
    rec = run_ml_experiment(engine.quick_baseline(), ws, store, source="baseline")
    after_experiment(rec, ws, store, engine, llm, X)
    if rec["status"] != "success":
        # даже быстрый бейзлайн не прошёл — пробуем самую надёжную модель
        h = {"summary": "Резервный бейзлайн: HistGradientBoosting", "implementation":
             {"kind": "registry", "model": "hist_gb", "params": {"max_iter": 200, "learning_rate": 0.1}}}
        rec = run_ml_experiment(h, ws, store, source="baseline")
        after_experiment(rec, ws, store, engine, llm, X)

    research_total = max(1.0, rc.remaining_research())
    tuned_models: set = set()
    tuning_on = os.environ.get("AGENT_DISABLE_TUNING", "0") != "1"

    def tune(budget: float) -> bool:
        """Тюнинг лучшей ещё не тюненной модели; True, если запущен эксперимент с новыми параметрами."""
        if not tuning_on or len(tuned_models) >= 2 or budget < 120:
            return False
        try:
            h, model = run_tuning(ws, store, budget, exclude=tuned_models)
            if model is None:
                tuned_models.add("__none__")
                tuned_models.add("__none2__")
            else:
                tuned_models.add(model)
        except Exception as e:  # noqa: BLE001
            log(f"Тюнинг упал: {type(e).__name__}: {short(e, 300)}")
            return False
        if h is None:
            return False
        rec = run_ml_experiment(h, ws, store, source="tune")
        after_experiment(rec, ws, store, engine, llm, X)
        return True

    while rc.research_continues():
        rc.iterations += 1
        if not tuned_models and rc.remaining_research() < 0.55 * research_total and rc.remaining_research() > 300:
            if tune(min(0.35 * rc.remaining_research(), 2700.0)):
                continue
        h, source = engine.next()
        if h is None:
            # гипотезы кончились, а время есть — тратим его на тюнинг, затем бэггинг тюненной модели
            if tune(min(0.5 * rc.remaining_research(), 3600.0)):
                continue
            log("Гипотезы закончились (или ни одна не укладывается во время) — завершаю исследование")
            break
        rec = run_ml_experiment(h, ws, store, source=source)
        after_experiment(rec, ws, store, engine, llm, X)
        if rec["status"] == "error" and rec.get("kind") == "code" and source == "llm":
            fixed = engine.repair(rec)
            if fixed is not None:
                rec = run_ml_experiment(fixed, ws, store, source="llm_fix")
                after_experiment(rec, ws, store, engine, llm, X)

    # 6. финал
    final_stage(ws, store, engine, llm, X)
    write_summary(ws, store, eda_summary)
    best = store.best()
    log(f"Готово за {fmt_duration(rc.elapsed())}. Лучший: {best['exp_id'] if best else '—'} "
        f"{metric.name}={fmt_num(best['local_cv_score']) if best else '—'}. Сабмит: {cfg.submission_path}")


def _emergency_submission(task_path: str) -> None:
    """Если всё упало до записи сабмита — кладём хотя бы sample_submission в правильном формате."""
    task_md = Path(task_path).resolve()
    out = task_md.parent / "submission.csv"
    if out.exists():
        return
    for p in sorted(task_md.parent.rglob("sample_submission*.csv")):
        shutil.copy(p, out)
        log(f"АВАРИЙНО: скопирован {p.name} → submission.csv")
        return


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    task_path = args[0] if args else "task.md"
    try:
        run(task_path)
    except Exception:  # noqa: BLE001
        log("ОШИБКА оркестратора:\n" + traceback.format_exc())
        try:
            _emergency_submission(task_path)
        except Exception:  # noqa: BLE001
            log("не удалось записать аварийный сабмит:\n" + traceback.format_exc())
    print("Агент завершил работу.", flush=True)


if __name__ == "__main__":
    main()
