"""ml_experiment.py — задача 4: проведение экспериментов и формирование сабмишена (харнесс).

Харнесс — фиксированный код, которому LLM не может «подыграть»: он режет одни и те же фолды,
сам считает метрику по OOF, ведёт историю экспериментов, выбирает лучший и пишет сабмит.
LLM (или fallback-лестница) решает только ЧТО обучать.

Виды экспериментов (hypothesis["implementation"]["kind"]):
    registry    — модель из model_zoo с параметрами (LLM передаёт любые params, loss и т.д.)
    code        — функция fit_predict, написанная LLM (запускается в отдельном процессе)
    blend       — взвешенный ансамбль OOF/test уже посчитанных экспериментов (веса — на OOF)
    postprocess — подбор весов классов на OOF для метрик по меткам (честная оценка через cross-fitting)

Совместимость с ранними форматами из чата команды:
    {"summary", "model_name", "params"}                         — формат первой версии ml_experiment
    {"statement", "interventions": [{"axis","action","params"}]} — формат hypothesis_001.json
Оба приводятся к единому виду normalize_hypothesis().

История — agent_work/experiment_history.json, запись на эксперимент:
    exp_id, timestamp, hypothesis_summary, kind, model, status, task_type, metric_name,
    local_cv_score, cv_std, fold_scores, delta_vs_best, is_new_best, oof_path, test_pred_path,
    duration_sec, error_message, reflection, source, signature, hypothesis
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import model_zoo
from agent_utils import fmt_num, log, read_json, short, tail, write_json
from task_metrics import Metric, evaluate, resolve_metric

AGENT_DIR = Path(__file__).resolve().parent
RUNNER = AGENT_DIR / "experiment_runner.py"
HISTORY_FILE = "experiment_history.json"


@dataclass
class Workspace:
    """Всё, что нужно харнессу: постановка, данные (мета), валидация, ресурсы, рабочая папка."""
    cfg: object               # TaskConfig
    work_dir: Path
    meta: dict
    plan: object              # ValidationPlan
    rc: object                # ResourceController
    y: np.ndarray
    test_ids: np.ndarray | None
    sample: pd.DataFrame | None
    metric: Metric

    @property
    def ctx(self) -> dict:
        return {"task_type": self.meta["task_type"], "n_classes": self.meta.get("n_classes") or 0,
                "positive_index": self.meta.get("positive_index", 1), "classes": self.meta.get("classes")}


# ============================================================================ история

class ExperimentStore:
    def __init__(self, work_dir: Path, metric: Metric):
        self.work_dir = Path(work_dir)
        self.path = self.work_dir / HISTORY_FILE
        self.metric = metric
        self.history: list[dict] = read_json(self.path, default=[]) or []

    def save(self) -> None:
        write_json(self.path, self.history)

    def next_id(self) -> str:
        return f"exp_{len(self.history):03d}"

    def successes(self) -> list[dict]:
        return [h for h in self.history if h.get("status") == "success" and h.get("local_cv_score") is not None]

    def best(self) -> dict | None:
        best = None
        for h in self.successes():
            if best is None or self.metric.is_better(h["local_cv_score"], best["local_cv_score"]):
                best = h
        return best

    def get(self, exp_id: str) -> dict | None:
        return next((h for h in self.history if h.get("exp_id") == exp_id), None)

    def signatures(self) -> set[str]:
        return {h.get("signature") for h in self.history if h.get("signature")}

    def add(self, rec: dict) -> dict:
        best = self.best()
        best_score = best["local_cv_score"] if best else None
        if rec.get("status") == "success":
            rec["delta_vs_best"] = round(self.metric.delta(rec["local_cv_score"], best_score), 6) if best else 0.0
            rec["is_new_best"] = best is None or self.metric.is_better(rec["local_cv_score"], best_score)
            noise = 0.25 * float(rec.get("cv_std") or (best or {}).get("cv_std") or 0.0)
            rec["within_noise"] = best is not None and abs(rec["delta_vs_best"]) <= noise
        else:
            rec["delta_vs_best"] = None
            rec["is_new_best"] = False
        rec["best_before"] = best_score
        self.history.append(rec)
        self.save()
        return rec

    def update(self, exp_id: str, **fields) -> None:
        rec = self.get(exp_id)
        if rec is not None:
            rec.update(fields)
            self.save()


# ============================================================================ гипотезы → спецификация

def normalize_hypothesis(h: dict) -> dict:
    """Привести гипотезу любого формата команды к виду {"summary", "implementation": {...}, ...}."""
    h = dict(h or {})
    impl = dict(h.get("implementation") or {})
    if not impl and h.get("model_name"):  # формат первой версии ml_experiment
        impl = {"kind": "registry", "model": h["model_name"], "params": h.get("params") or {}}
    if not impl and h.get("interventions"):  # формат hypothesis_001.json
        params, features, model = {}, {}, None
        for it in h["interventions"]:
            axis = str(it.get("axis", "")).lower()
            if axis in ("model_architecture", "model", "algorithm", "hyperparameters"):
                model = model or it.get("action")
                params.update(it.get("params") or {})
            elif axis in ("data_processing", "features", "feature_engineering", "preprocessing"):
                p = it.get("params") or {}
                drop = p.get("drop_columns") or []
                features["drop_columns"] = list(drop)
                if p.get("frequency_encoding"):
                    features["frequency_encoding"] = True
            # axis == "validation" игнорируется: схема валидации фиксирована харнессом
        impl = {"kind": "registry", "model": model, "params": params, "features": features}
    kind = impl.get("kind") or ("code" if h.get("code") else "registry")
    impl["kind"] = kind
    if kind == "registry":
        impl["model"] = model_zoo.normalize_model_name(impl.get("model")) or impl.get("model")
        params = dict(impl.get("params") or {})
        # служебные параметры, которые харнесс выставляет сам
        for k in ("random_seed", "random_state", "thread_count", "n_jobs", "verbose", "silent", "logging_level"):
            params.pop(k, None)
        if str(params.get("auto_class_weights", "")).lower() == "balanced" or \
                str(params.get("class_weight", params.get("class_weights", ""))).lower() == "balanced":
            params.pop("auto_class_weights", None)
            if isinstance(params.get("class_weight"), str):
                params.pop("class_weight", None)
            if isinstance(params.get("class_weights"), str):
                params.pop("class_weights", None)
            params["class_balance"] = True
        impl["params"] = params
        impl["features"] = dict(impl.get("features") or {})
    if kind == "code":
        impl["code"] = impl.get("code") or h.get("code")
    h["implementation"] = impl
    h["summary"] = h.get("summary") or h.get("hypothesis_summary") or h.get("statement") or f"{kind} {impl.get('model', '')}"
    return h


def spec_signature(impl: dict) -> str:
    """Подпись эксперимента для поиска дублей."""
    kind = impl.get("kind")
    if kind == "registry":
        payload = {"k": kind, "m": impl.get("model"), "p": impl.get("params") or {}, "f": impl.get("features") or {}}
    elif kind == "code":
        code = "\n".join(ln.rstrip() for ln in (impl.get("code") or "").splitlines() if ln.strip() and not ln.strip().startswith("#"))
        payload = {"k": kind, "c": hashlib.md5(code.encode()).hexdigest()}
    elif kind == "blend":
        payload = {"k": kind, "e": sorted(impl.get("exp_ids") or [])}
    else:
        payload = {"k": kind, "b": impl.get("base_exp_id")}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def family_of(impl: dict) -> str:
    kind = impl.get("kind")
    return f"registry:{impl.get('model')}" if kind == "registry" else str(kind)


def work_units_of(impl: dict) -> float:
    """Объём работы эксперимента для оценки длительности (см. model_zoo.work_units)."""
    if impl.get("kind") == "registry":
        return model_zoo.work_units(impl.get("model"), impl.get("params"), impl.get("features"))
    return 1.0


# ============================================================================ запуск

def _subprocess_env(n_threads: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(AGENT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[var] = str(n_threads)  # sklearn/HGB, numpy и бустинги не должны брать больше потоков, чем есть CPU
    return env


def _run_subprocess(exp_dir: Path, timeout: float, n_threads: int) -> tuple[int, str]:
    try:
        proc = subprocess.run([sys.executable, "-u", str(RUNNER), str(exp_dir)], capture_output=True, text=True,
                              timeout=timeout, env=_subprocess_env(n_threads), cwd=str(exp_dir))
        out = (proc.stdout or "") + ("\n[stderr]\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out
    except subprocess.TimeoutExpired as e:
        out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
        return -9, out + f"\n[таймаут {timeout:.0f}с — процесс остановлен]"


def run_ml_experiment(hypothesis: dict, ws: Workspace, store: ExperimentStore, source: str = "llm",
                      final_stage: bool = False) -> dict:
    """Провести эксперимент по гипотезе и записать результат в историю. Возвращает запись истории.

    Для kind=registry/code — CV по фиксированным фолдам в отдельном процессе с таймаутом
    от ResourceController. Для blend/postprocess — быстрые операции над сохранёнными OOF.
    """
    h = normalize_hypothesis(hypothesis)
    impl = h["implementation"]
    kind = impl["kind"]
    exp_id = store.next_id()
    exp_dir = ws.work_dir / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "exp_id": exp_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hypothesis_summary": short(h["summary"], 300), "kind": kind, "model": impl.get("model"),
        "source": source, "task_type": ws.cfg.task_type, "metric_name": ws.metric.name,
        "signature": spec_signature(impl), "status": "error", "local_cv_score": None, "cv_std": None,
        "fold_scores": [], "oof_path": None, "test_pred_path": None, "model_artifact": None,
        "duration_sec": None, "error_message": None, "reflection": None,
        "hypothesis": {k: v for k, v in h.items() if k != "code"},
    }
    if kind == "code":
        rec["hypothesis"]["implementation"] = {k: v for k, v in impl.items() if k != "code"}
    write_json(exp_dir / "hypothesis.json", h)
    t0 = time.time()
    try:
        if kind in ("registry", "code"):
            _run_cv(impl, ws, exp_dir, rec, final_stage)
        elif kind == "blend":
            _run_blend(impl, ws, store, exp_dir, rec)
        elif kind == "postprocess":
            _run_postprocess(impl, ws, store, exp_dir, rec)
        else:
            raise ValueError(f"неизвестный kind={kind}")
    except Exception as e:  # noqa: BLE001
        rec["status"] = "error"
        rec["error_message"] = short(f"{type(e).__name__}: {e}", 1500)
    rec["duration_sec"] = round(time.time() - t0, 1)
    ws.rc.record_duration(family_of(impl), rec["duration_sec"], rec["status"], work_units_of(impl))
    store.add(rec)
    msg = (f"{exp_id} [{kind}{':' + str(impl.get('model')) if impl.get('model') else ''}] {rec['status']}: "
           f"{ws.metric.name}={fmt_num(rec['local_cv_score'])}")
    if rec["status"] == "success":
        msg += f" (Δ={fmt_num(rec['delta_vs_best'])}, {'НОВЫЙ ЛУЧШИЙ' if rec['is_new_best'] else 'не лучше'})"
    else:
        msg += f" — {short(rec['error_message'], 200)}"
    log(msg + f" за {rec['duration_sec']}с")
    return rec


def _run_cv(impl: dict, ws: Workspace, exp_dir: Path, rec: dict, final_stage: bool) -> None:
    kind = impl["kind"]
    if kind == "registry":
        if impl.get("model") not in model_zoo.available_models():
            raise ValueError(f"модель '{impl.get('model')}' недоступна. Доступны: {list(model_zoo.available_models())}")
    else:
        if not impl.get("code"):
            raise ValueError("kind=code, но код не передан")
        (exp_dir / "model_code.py").write_text(impl["code"], encoding="utf-8")
    est = ws.rc.estimate(family_of(impl), work_units_of(impl))
    timeout = ws.rc.timeout_for(est, final_stage=final_stage)
    deadline = time.time() + timeout - 5
    spec = {
        "work_dir": str(ws.work_dir), "kind": kind, "model": impl.get("model"), "params": impl.get("params"),
        "features": impl.get("features"), "n_folds": ws.plan.n_splits, "n_threads": ws.rc.n_threads,
        "seed": 42, "deadline_ts": deadline, "mode": "cv",
        "test_agg": "last" if ws.plan.scheme == "time_forward" else "mean",
    }
    write_json(exp_dir / "spec.json", spec)
    log(f"{rec['exp_id']}: старт {kind}{':' + str(impl.get('model')) if impl.get('model') else ''} "
        f"(оценка {est:.0f}с, таймаут {timeout:.0f}с) — {short(rec['hypothesis_summary'], 120)}")
    code, out = _run_subprocess(exp_dir, timeout, ws.rc.n_threads)
    (exp_dir / "stdout.log").write_text(tail(out, 20000), encoding="utf-8")
    res = read_json(exp_dir / "result.json", default=None)
    if res is None:
        rec["status"] = "timeout" if code == -9 else "error"
        rec["error_message"] = tail(out, 1500) or f"процесс завершился с кодом {code}"
        return
    rec["fold_scores"] = res.get("fold_scores", [])
    rec["fold_seconds"] = res.get("fold_seconds", [])
    if res.get("status") != "success":
        rec["status"] = res.get("status", "error")
        rec["error_message"] = short(res.get("error_message"), 1000)
        rec["traceback"] = res.get("traceback")
        return
    rec.update(status="success", local_cv_score=res["score"], cv_std=res.get("fold_std"),
               fold_mean=res.get("fold_mean"), oof_coverage=res.get("oof_coverage"),
               oof_path=str(exp_dir / "oof.npy"), test_pred_path=str(exp_dir / "test.npy"))


# ============================================================================ бленд и постобработка

def run_tuning(ws: Workspace, store: ExperimentStore, budget_sec: float, exclude: set | None = None) -> tuple[dict | None, str | None]:
    """Подбор гиперпараметров лучшей модели реестра (optuna / случайный поиск) в отдельном процессе.

    Возвращает гипотезу kind=registry с найденными параметрами — её честно оценивает обычный эксперимент.
    """
    regs = [h for h in store.successes() if h.get("kind") == "registry" and h.get("model") in model_zoo.TUNABLE
            and h.get("model") not in (exclude or set())]
    if not regs:
        return None, None
    base = max(regs, key=lambda h: h["local_cv_score"]) if store.metric.greater_is_better else \
        min(regs, key=lambda h: h["local_cv_score"])
    impl = (base.get("hypothesis") or {}).get("implementation") or {}
    params = {k: v for k, v in (impl.get("params") or {}).items() if k != "seed_offset"}
    tdir = ws.work_dir / "tuning" / f"tune_{base['exp_id']}"
    tdir.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + budget_sec
    spec = {
        "work_dir": str(ws.work_dir), "kind": "registry", "model": impl.get("model"), "params": params,
        "features": impl.get("features") or {}, "n_folds": ws.plan.n_splits, "n_threads": ws.rc.n_threads,
        "seed": 42, "deadline_ts": deadline, "mode": "tune", "time_forward": ws.plan.scheme == "time_forward",
    }
    write_json(tdir / "spec.json", spec)
    log(f"Тюнинг {impl.get('model')} от {base['exp_id']} ({ws.metric.name}={fmt_num(base['local_cv_score'])}), "
        f"бюджет {budget_sec:.0f}с")
    code, out = _run_subprocess(tdir, budget_sec + 120, ws.rc.n_threads)
    (tdir / "stdout.log").write_text(tail(out, 20000), encoding="utf-8")
    res = read_json(tdir / "tune_result.json", default=None)
    if not res or not res.get("best_params"):
        log(f"Тюнинг не дал результата: {short(tail(out, 600), 400)}")
        return None, impl.get("model")
    log(f"Тюнинг: {res.get('n_trials')} проб, лучший скор на подвыборке фолдов {fmt_num(res.get('best_score'))}")
    if res["best_params"] == params:
        log("Тюнинг не нашёл параметров лучше исходных")
        return None, impl.get("model")
    return {"summary": f"{impl.get('model')} с параметрами из тюнинга ({res.get('n_trials')} проб) поверх {base['exp_id']}",
            "statement": "подобранные гиперпараметры лучшей модели", "implementation":
            {"kind": "registry", "model": impl.get("model"), "params": res["best_params"],
             "features": impl.get("features") or {}}}, impl.get("model")


def _load_preds(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.load(rec["oof_path"]), np.load(rec["test_pred_path"])


def _mask(oof: np.ndarray) -> np.ndarray:
    return ~np.isnan(oof).any(axis=1) if oof.ndim == 2 else ~np.isnan(oof)


def _blend_score(ws: Workspace, preds: list[np.ndarray], w: np.ndarray, y: np.ndarray) -> float:
    p = sum(wi * pi for wi, pi in zip(w, preds)) / max(w.sum(), 1e-12)
    return evaluate(ws.metric, y, p, ws.ctx)


def _run_blend(impl: dict, ws: Workspace, store: ExperimentStore, exp_dir: Path, rec: dict) -> None:
    ids = impl.get("exp_ids") or []
    if ids == "auto" or not ids:
        ids = pick_blend_candidates(store)
    recs = [store.get(i) for i in ids]
    recs = [r for r in recs if r and r.get("status") == "success" and r.get("oof_path")]
    if len(recs) < 2:
        raise ValueError(f"для бленда нужно ≥2 успешных эксперимента с OOF, есть: {[r['exp_id'] for r in recs]}")
    oofs, tests = zip(*[_load_preds(r) for r in recs])
    mask = np.logical_and.reduce([_mask(o) for o in oofs])
    y = ws.y[mask]
    preds = [o[mask] for o in oofs]
    w = np.ones(len(preds))
    best = _blend_score(ws, preds, w, y)
    grid = np.linspace(0, 1, 11)
    for _ in range(3):  # покоординатный подбор весов
        improved = False
        for i in range(len(w)):
            for g in grid:
                cand = w.copy()
                cand[i] = g * len(w)
                if cand.sum() == 0:
                    continue
                s = _blend_score(ws, preds, cand, y)
                if ws.metric.is_better(s, best):
                    best, w, improved = s, cand, True
        if not improved:
            break
    w = w / w.sum()
    oof = sum(wi * o for wi, o in zip(w, oofs))
    test = sum(wi * t for wi, t in zip(w, tests))
    np.save(exp_dir / "oof.npy", oof)
    np.save(exp_dir / "test.npy", test)
    rec.update(status="success", local_cv_score=round(float(best), 6), oof_path=str(exp_dir / "oof.npy"),
               test_pred_path=str(exp_dir / "test.npy"), components=[r["exp_id"] for r in recs],
               weights=[round(float(x), 4) for x in w])
    rec["hypothesis_summary"] = short(f"{rec['hypothesis_summary']} | веса: " +
                                      ", ".join(f"{r['exp_id']}={x:.2f}" for r, x in zip(recs, w)), 300)


def pick_blend_candidates(store: ExperimentStore, k: int = 5, per_family: int = 3) -> list[str]:
    """Лучшие успешные эксперименты (кроме блендов и постобработки): до k штук, не больше per_family
    от одной модели — чтобы в ансамбль попали и непохожие модели, и бэггинг лучшей по сидам."""
    ok = [h for h in store.successes() if h.get("kind") in ("registry", "code")]
    ok.sort(key=lambda h: h["local_cv_score"], reverse=store.metric.greater_is_better)
    chosen, fams = [], {}
    for h in ok:
        fam = (h.get("kind"), h.get("model"))
        if fams.get(fam, 0) >= per_family:
            continue
        chosen.append(h["exp_id"])
        fams[fam] = fams.get(fam, 0) + 1
        if len(chosen) >= k:
            break
    return chosen


def _fit_class_weights(p: np.ndarray, y: np.ndarray, ws: Workspace) -> np.ndarray:
    k = p.shape[1]
    w = np.ones(k)
    grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 15))

    def sc(wv):
        return evaluate(ws.metric, y, p * wv, ws.ctx)

    best = sc(w)
    for _ in range(2):
        for c in range(k):
            for g in grid:
                cand = w.copy()
                cand[c] = g
                s = sc(cand)
                if ws.metric.is_better(s, best):
                    best, w = s, cand
    return w


def _run_postprocess(impl: dict, ws: Workspace, store: ExperimentStore, exp_dir: Path, rec: dict) -> None:
    if not ws.cfg.is_classification or ws.metric.needs != "label":
        raise ValueError("постобработка весами классов имеет смысл только для классификации с метрикой по меткам")
    base = store.get(impl.get("base_exp_id") or "") or store.best()
    if not base or base.get("status") != "success":
        raise ValueError("нет успешного базового эксперимента для постобработки")
    if base.get("kind") == "postprocess":
        raise ValueError("базовый эксперимент уже постобработан")
    oof, test = _load_preds(base)
    mask = _mask(oof)
    p, y = oof[mask], ws.y[mask]
    # честная оценка: веса подбираются на одной половине OOF, проверяются на другой
    rng = np.random.default_rng(0)
    half = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        half[rng.permutation(idx)[: len(idx) // 2]] = True
    adj = np.empty_like(p)
    for a, b in ((half, ~half), (~half, half)):
        wv = _fit_class_weights(p[a], y[a], ws)
        adj[b] = p[b] * wv
    honest = evaluate(ws.metric, y, adj, ws.ctx)
    w_all = _fit_class_weights(p, y, ws)
    new_oof = oof.copy()
    new_oof[mask] = p * w_all
    new_oof[mask] /= new_oof[mask].sum(axis=1, keepdims=True)
    new_test = test * w_all
    new_test /= new_test.sum(axis=1, keepdims=True)
    np.save(exp_dir / "oof.npy", new_oof)
    np.save(exp_dir / "test.npy", new_test)
    rec.update(status="success", local_cv_score=round(float(honest), 6), oof_path=str(exp_dir / "oof.npy"),
               test_pred_path=str(exp_dir / "test.npy"), base_exp_id=base["exp_id"],
               class_weights=[round(float(x), 3) for x in w_all])
    rec["hypothesis_summary"] = short(f"{rec['hypothesis_summary']} | база {base['exp_id']} "
                                      f"({fmt_num(base['local_cv_score'])}), честная оценка cross-fit", 300)


# ============================================================================ refit

def refit_on_full(rec: dict, ws: Workspace) -> bool:
    """Переобучить эксперимент на всём train (для временной валидации, где фолды видели не все периоды)."""
    impl = rec["hypothesis"]["implementation"]
    if rec.get("kind") not in ("registry", "code"):
        return False
    exp_dir = Path(rec["oof_path"]).parent
    per_fold = float(np.mean(rec.get("fold_seconds") or [rec.get("duration_sec") or 60]))
    est = per_fold * 1.6
    if ws.rc.remaining() < est * 1.3 + 30:
        log(f"refit {rec['exp_id']}: не хватает времени (нужно ~{est:.0f}с)")
        return False
    spec = read_json(exp_dir / "spec.json")
    spec.update(mode="refit", deadline_ts=time.time() + ws.rc.remaining() - 20)
    write_json(exp_dir / "spec.json", spec)
    log(f"refit {rec['exp_id']} на всём train (~{est:.0f}с)")
    code, out = _run_subprocess(exp_dir, max(30.0, ws.rc.remaining() - 20), ws.rc.n_threads)
    (exp_dir / "stdout_refit.log").write_text(tail(out, 10000), encoding="utf-8")
    res = read_json(exp_dir / "result_refit.json", default={})
    if res.get("status") == "success" and (exp_dir / "test_refit.npy").exists():
        rec["test_refit_path"] = str(exp_dir / "test_refit.npy")
        return True
    log(f"refit {rec['exp_id']} не удался: {short(res.get('error_message') or out, 300)}")
    return False


# ============================================================================ сабмит

def predictions_to_frame(test_pred: np.ndarray, ws: Workspace) -> pd.DataFrame:
    cfg, meta = ws.cfg, ws.meta
    cols = list(cfg.submission_columns)
    id_col = cfg.id_col if cfg.id_col in cols else cols[0]
    ids = ws.test_ids if ws.test_ids is not None else (ws.sample[id_col].values if ws.sample is not None
                                                        else np.arange(len(test_pred)))
    out = pd.DataFrame({id_col: ids})
    fmt = cfg.prediction_format
    if cfg.task_type == "regression":
        lo, hi = meta.get("target_min"), meta.get("target_max")
        vals = np.asarray(test_pred, float)
        if lo is not None and hi is not None:
            vals = np.clip(vals, lo, hi)
        out[cfg.target if cfg.target in cols else cols[-1]] = vals
    else:
        classes = meta["classes"]
        if fmt == "proba_per_class":
            for i, c in enumerate(classes):
                col = next((x for x in cols if str(x) == str(c)), None)
                if col is not None:
                    out[col] = test_pred[:, i]
        elif fmt == "proba_positive":
            out[cfg.target if cfg.target in cols else cols[-1]] = test_pred[:, int(meta.get("positive_index", 1))]
        else:
            labels = np.asarray(classes, dtype=object)[test_pred.argmax(axis=1)]
            out[cfg.target if cfg.target in cols else cols[-1]] = labels
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[cols]


def validate_submission(df: pd.DataFrame, ws: Workspace) -> list[str]:
    problems = []
    cfg = ws.cfg
    n_expected = len(ws.test_ids) if ws.test_ids is not None else (len(ws.sample) if ws.sample is not None else len(df))
    if len(df) != n_expected:
        problems.append(f"строк {len(df)}, ожидалось {n_expected}")
    if list(df.columns) != list(cfg.submission_columns):
        problems.append(f"колонки {list(df.columns)} != {cfg.submission_columns}")
    if df.isna().any().any():
        problems.append("есть пропуски")
    id_col = cfg.id_col if cfg.id_col in df.columns else df.columns[0]
    if df[id_col].duplicated().any():
        problems.append("повторяющиеся id")
    if ws.test_ids is not None and not np.array_equal(df[id_col].values, ws.test_ids):
        problems.append("порядок id не совпадает с test")
    num = df.drop(columns=[id_col]).select_dtypes("number")
    if len(num.columns) and not np.isfinite(num.values).all():
        problems.append("есть inf")
    if cfg.is_classification and cfg.prediction_format == "label":
        allowed = {str(c) for c in ws.meta["classes"]}
        bad = ~df[cfg.target].astype(str).isin(allowed)
        if bad.any():
            problems.append(f"метки вне списка классов: {df[cfg.target][bad].unique()[:5]}")
    return problems


def _write_submission(df: pd.DataFrame, ws: Workspace) -> str:
    path = Path(ws.cfg.submission_path)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return str(path)


def generate_submission(ws: Workspace, store: ExperimentStore, exp_id: str | None = None) -> dict:
    """Записать submission.csv по лучшему (или указанному) эксперименту. Возвращает статус."""
    rec = store.get(exp_id) if exp_id else store.best()
    if rec is None:
        return {"status": "error", "message": "нет успешных экспериментов"}
    path = rec.get("test_refit_path") or rec["test_pred_path"]
    test_pred = np.load(path)
    df = predictions_to_frame(test_pred, ws)
    problems = validate_submission(df, ws)
    if problems:
        log(f"Сабмит по {rec['exp_id']} не прошёл проверку: {problems}")
        return {"status": "error", "message": "; ".join(problems)}
    out = _write_submission(df, ws)
    write_json(ws.work_dir / "submission_source.json",
               {"exp_id": rec["exp_id"], "score": rec["local_cv_score"], "metric": ws.metric.name,
                "refit": bool(rec.get("test_refit_path")), "path": out})
    log(f"submission.csv ← {rec['exp_id']} ({ws.metric.name}={fmt_num(rec['local_cv_score'])}"
        f"{', refit' if rec.get('test_refit_path') else ''})")
    return {"status": "success", "submission_path": out, "exp_id": rec["exp_id"]}


def write_fallback_submission(ws: Workspace) -> str:
    """Тривиальный, но валидный сабмит в первые секунды запуска: среднее / самый частый класс / априорные вероятности."""
    n_test = len(ws.test_ids) if ws.test_ids is not None else len(ws.sample)
    if ws.cfg.task_type == "regression":
        if ws.metric.name in ("mae", "medae", "mape", "wape", "smape"):
            pred = np.full(n_test, float(np.median(ws.y)))
        else:
            pred = np.full(n_test, float(np.mean(ws.y)))
    else:
        k = int(ws.meta["n_classes"])
        prior = np.bincount(ws.y, minlength=k) / len(ws.y)
        pred = np.tile(prior, (n_test, 1))
    df = predictions_to_frame(pred, ws)
    problems = validate_submission(df, ws)
    if problems:
        log(f"fallback-сабмит: замечания {problems}")
    out = _write_submission(df, ws)
    log(f"Записан fallback-сабмит ({'среднее' if ws.cfg.task_type == 'regression' else 'априорные классы'}): {out}")
    return out


# ============================================================================ совместимость

def get_best_model_summary(store: ExperimentStore) -> str:
    b = store.best()
    if not b:
        return "успешных экспериментов пока нет"
    return f"{b['exp_id']} ({b['kind']}:{b.get('model') or ''}) {store.metric.name}={fmt_num(b['local_cv_score'])} — {short(b['hypothesis_summary'], 160)}"


def load_metric_from_config(cfg_dict: dict) -> Metric:
    m = resolve_metric(cfg_dict.get("metric") or cfg_dict.get("metric_name"), cfg_dict.get("task_type"))
    if m is None:
        raise ValueError(f"неизвестная метрика: {cfg_dict.get('metric')}")
    return m
