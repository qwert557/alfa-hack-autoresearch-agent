"""experiment_runner.py — исполнитель одного эксперимента в отдельном процессе.

    python experiment_runner.py <exp_dir>

Читает <exp_dir>/spec.json, данные из agent_work/cache/data.pkl и фолды из agent_work/folds.npz,
получает функцию fit_predict (из реестра model_zoo или из кода LLM <exp_dir>/model_code.py)
и прогоняет её по всем фолдам. Таргет валидации функция не видит — метрику считает только раннер.

Результат:
    oof.npy        — OOF-предсказания (NaN там, где строка не была в валидации, напр. ранние периоды)
    test.npy       — предсказания на test, усреднённые по фолдам (при временной валидации — из последнего
                     фолда, обученного на самых свежих периодах)
    result.json    — status, fold_scores, score, duration, ошибка с трейсбеком
Режим refit (spec["mode"] == "refit"): обучение на всём train, только test_refit.npy.

Отдельный процесс нужен, чтобы падение/зависание/утечка памяти в коде LLM не роняли агента,
а таймаут гарантированно соблюдался.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent_utils import read_json, tail, write_json  # noqa: E402
import model_zoo  # noqa: E402
from task_metrics import evaluate, resolve_metric  # noqa: E402


def _load_fit_predict(spec: dict, exp_dir: Path):
    kind = spec.get("kind")
    if kind == "registry":
        return model_zoo.make_fit_predict(spec["model"], spec.get("params") or {}, spec.get("features") or {})
    if kind == "code":
        code_path = exp_dir / "model_code.py"
        code = code_path.read_text(encoding="utf-8")
        ns: dict = {"__name__": "llm_model_code", "__file__": str(code_path)}
        exec(compile(code, str(code_path), "exec"), ns)  # noqa: S102 — код от своей LLM, в отдельном процессе
        fp = ns.get("fit_predict")
        if not callable(fp):
            raise ValueError("в коде нет функции fit_predict(X_train, y_train, X_val, X_test, ctx)")
        return fp
    raise ValueError(f"раннер не умеет kind={kind}")


def _check_pred(pred, n: int, ctx: dict, what: str) -> np.ndarray:
    arr = np.asarray(pred)
    clf = ctx["task_type"] != "regression"
    k = int(ctx.get("n_classes") or 0)
    if clf:
        if arr.ndim == 1:
            if arr.dtype.kind in "iu" or (arr.dtype.kind == "f" and np.all(np.mod(arr[~np.isnan(arr)], 1) == 0) and k > 2):
                lab = arr.astype(int)
                if lab.min() < 0 or lab.max() >= k:
                    raise ValueError(f"{what}: метки вне диапазона 0..{k - 1}")
                arr = np.eye(k)[lab]
            elif k == 2:
                arr = np.c_[1 - arr.astype(float), arr.astype(float)]
            else:
                raise ValueError(f"{what}: для {k} классов нужна матрица вероятностей (n, {k})")
        arr = arr.astype(float)
        if arr.shape != (n, k):
            raise ValueError(f"{what}: ожидалась форма {(n, k)}, получено {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{what}: есть NaN/inf в вероятностях")
        arr = np.clip(arr, 0, None)
        s = arr.sum(axis=1, keepdims=True)
        s[s == 0] = 1
        return arr / s
    arr = arr.astype(float).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{what}: ожидалось {n} значений, получено {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{what}: есть NaN/inf в предсказаниях")
    return arr


def main(exp_dir: Path) -> int:
    t_start = time.time()
    spec = read_json(exp_dir / "spec.json")
    work = Path(spec["work_dir"])
    result: dict = {"status": "error", "fold_scores": [], "fold_seconds": []}
    try:
        data = pd.read_pickle(work / "cache" / "data.pkl")
        X, y, X_test, meta = data["X"], np.asarray(data["y"]), data["X_test"], data["meta"]
        ctx = {
            "task_type": meta["task_type"], "n_classes": meta.get("n_classes") or 0,
            "classes": meta.get("classes"), "cat_features": list(meta["cat_features"]),
            "num_features": list(meta["num_features"]), "n_threads": int(spec.get("n_threads", 1)),
            "seed": int(spec.get("seed", 42)), "metric": meta["metric"],
            "positive_index": meta.get("positive_index", 1),
        }
        metric = resolve_metric(meta["metric"], meta["task_type"])
        fit_predict = _load_fit_predict(spec, exp_dir)
        deadline = float(spec.get("deadline_ts", time.time() + 1e9))

        if spec.get("mode") == "refit":
            ctx["time_budget_sec"] = max(10.0, deadline - time.time())
            _, test_pred = fit_predict(X.copy(), y.copy(), X_test.copy(), X_test.copy(), ctx)
            test_pred = _check_pred(test_pred, len(X_test), ctx, "test_pred")
            np.save(exp_dir / "test_refit.npy", test_pred)
            result.update(status="success", duration_sec=round(time.time() - t_start, 2))
            write_json(exp_dir / "result_refit.json", result)
            return 0

        if spec.get("mode") == "tune":
            return _tune(spec, exp_dir, X, y, X_test, ctx, metric, work, deadline, t_start)

        folds_npz = np.load(work / "folds.npz")
        n_folds = int(spec["n_folds"])
        clf = ctx["task_type"] != "regression"
        k = int(ctx["n_classes"]) if clf else 0
        oof = np.full((len(X), k), np.nan) if clf else np.full(len(X), np.nan)
        test_acc = np.zeros((len(X_test), k)) if clf else np.zeros(len(X_test))
        fold_secs = []
        for f in range(n_folds):
            tr, va = folds_npz[f"train_{f}"], folds_npz[f"val_{f}"]
            t0 = time.time()
            remaining_time = deadline - t0
            ctx["time_budget_sec"] = max(10.0, remaining_time / max(1, n_folds - f))
            val_pred, test_pred = fit_predict(X.iloc[tr].copy(), y[tr].copy(), X.iloc[va].copy(), X_test.copy(), ctx)
            val_pred = _check_pred(val_pred, len(va), ctx, "val_pred")
            test_pred = _check_pred(test_pred, len(X_test), ctx, "test_pred")
            oof[va] = val_pred
            if spec.get("test_agg") == "last":
                test_acc = test_pred  # временная валидация: последний фолд обучен на самых свежих данных
            else:
                test_acc += test_pred / n_folds
            score = evaluate(metric, y[va], val_pred, ctx)
            fold_secs.append(round(time.time() - t0, 2))
            result["fold_scores"].append(round(score, 6))
            result["fold_seconds"] = fold_secs
            print(f"fold {f + 1}/{n_folds}: {meta['metric']}={score:.5f} ({fold_secs[-1]:.1f}s)", flush=True)
            left = n_folds - f - 1
            if left and time.time() + np.mean(fold_secs) * left * 1.05 > deadline:
                result.update(status="timeout", duration_sec=round(time.time() - t_start, 2),
                              error_message=(f"не успеваем: фолд в среднем {np.mean(fold_secs):.0f}с, осталось {left} "
                                             f"фолдов, до дедлайна {max(0, deadline - time.time()):.0f}с. "
                                             f"Нужна более быстрая конфигурация."))
                write_json(exp_dir / "result.json", result)
                return 3
        mask = ~np.isnan(oof).any(axis=1) if clf else ~np.isnan(oof)
        overall = evaluate(metric, y[mask], oof[mask], ctx)
        np.save(exp_dir / "oof.npy", oof)
        np.save(exp_dir / "test.npy", test_acc)
        result.update(status="success", score=round(float(overall), 6),
                      fold_mean=round(float(np.mean(result["fold_scores"])), 6),
                      fold_std=round(float(np.std(result["fold_scores"])), 6),
                      duration_sec=round(time.time() - t_start, 2), oof_coverage=round(float(mask.mean()), 4))
        write_json(exp_dir / "result.json", result)
        return 0
    except Exception as e:  # noqa: BLE001
        result.update(status="error", error_type=type(e).__name__, error_message=f"{type(e).__name__}: {e}",
                      traceback=tail(traceback.format_exc(), 4000), duration_sec=round(time.time() - t_start, 2))
        write_json(exp_dir / ("result_refit.json" if spec.get("mode") == "refit" else "result.json"), result)
        return 1


def _tune(spec, exp_dir, X, y, X_test, ctx, metric, work, deadline, t_start) -> int:
    """Подбор гиперпараметров модели реестра по фолдам плана (не больше 2 фолдов) до дедлайна.

    Пишет tune_result.json: лучшие параметры и скор. Честную оценку лучших параметров харнесс затем
    получает обычным экспериментом на всех фолдах.
    """
    name = spec["model"]
    base = dict(spec.get("params") or {})
    features = spec.get("features") or {}
    folds_npz = np.load(work / "folds.npz")
    n_all = int(spec["n_folds"])
    fold_ids = list(range(n_all))[-2:] if spec.get("time_forward") else list(range(min(2, n_all)))
    folds = [(folds_npz[f"train_{f}"], folds_npz[f"val_{f}"]) for f in fold_ids]
    sign = 1.0 if metric.greater_is_better else -1.0
    trials_log: list[dict] = []
    best = {"score": None, "params": base}
    seen = set()

    def evaluate_params(params: dict) -> float:
        fp = model_zoo.make_fit_predict(name, params, features)
        scores = []
        for tr, va in folds:
            ctx["time_budget_sec"] = max(10.0, (deadline - time.time()) / 2)
            val_pred, _ = fp(X.iloc[tr].copy(), y[tr].copy(), X.iloc[va].copy(), X.iloc[va[:5]].copy(), ctx)
            val_pred = _check_pred(val_pred, len(va), ctx, "val_pred")
            scores.append(evaluate(metric, y[va], val_pred, ctx))
        return float(np.mean(scores))

    def record(params: dict, score: float, sec: float) -> None:
        trials_log.append({"params": params, "score": round(score, 6), "sec": round(sec, 1)})
        if best["score"] is None or sign * score > sign * best["score"]:
            best.update(score=score, params=params)
        write_json(exp_dir / "tune_result.json", {"status": "success", "model": name, "best_params": best["params"],
                                                  "best_score": best["score"], "n_trials": len(trials_log),
                                                  "trials": trials_log[-50:]})
        print(f"trial {len(trials_log)}: {score:.5f} ({sec:.0f}s) best={best['score']:.5f}", flush=True)

    t0 = time.time()
    record(base, evaluate_params(base), time.time() - t0)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=int(spec.get("seed", 42)), n_startup_trials=6)
        study = optuna.create_study(direction="maximize" if metric.greater_is_better else "minimize", sampler=sampler)
        new_trial = study.ask
        tell = study.tell
    except ImportError:
        rng = np.random.default_rng(int(spec.get("seed", 42)))
        new_trial = lambda: model_zoo.RandomTrial(rng)  # noqa: E731
        tell = None
    durations = [time.time() - t0]
    while time.time() + 1.3 * float(np.median(durations)) < deadline and len(trials_log) < 200:
        trial = new_trial()
        params = model_zoo.suggest_params(name, trial, base)
        key = repr(sorted(params.items()))
        if key in seen:
            if tell:
                tell(trial, state=optuna.trial.TrialState.PRUNED)
            continue
        seen.add(key)
        t1 = time.time()
        try:
            score = evaluate_params(params)
        except Exception as e:  # noqa: BLE001
            print(f"trial failed: {type(e).__name__}: {e}", flush=True)
            if tell:
                tell(trial, state=optuna.trial.TrialState.FAIL)
            durations.append(time.time() - t1)
            continue
        durations.append(time.time() - t1)
        if tell:
            tell(trial, score)
        record(params, score, durations[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
