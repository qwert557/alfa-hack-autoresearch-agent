"""ml_analysis.py — задача 5: анализ ошибок экспериментов и рефлексия.

    get_experiment_history(...)     — компактная история для промпта (что пробовали, что вышло)
    analyze_experiment_errors(...)  — разбор OOF-ошибок эксперимента, данные → markdown + факты
    write_reflection(...)           — рефлексия: LLM по фактам анализа; без LLM — по правилам

Анализ строится по данным, а не шаблонным текстом:
  классификация — F1/precision/recall по классам и их изменение относительно лучшего эксперимента,
                  главные пары путаницы (истина → предсказание), доля ошибок по сегментам категорий;
  регрессия     — метрика, смещение (средний остаток), ошибка по децилям таргета (недооценка хвоста?),
                  доля квадратичной ошибки от худшего 1% строк, худшие сегменты категориальных признаков.
markdown-таблицы собираются без tabulate (его нет в requirements.txt).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import prompts
from agent_utils import fmt_num, md_table, read_json, short
from task_metrics import evaluate


# ============================================================================ история

def history_rows(history: list[dict], last_n: int = 12) -> list[dict]:
    rows = []
    for h in history[-last_n:]:
        rows.append({
            "exp_id": h.get("exp_id"), "kind": f"{h.get('kind')}:{h.get('model') or ''}".rstrip(":"),
            "summary": short(h.get("hypothesis_summary"), 110), "status": h.get("status"),
            "score": h.get("local_cv_score"),
            "delta": (f"{fmt_num(h.get('delta_vs_best'))} (шум)" if h.get("within_noise") else h.get("delta_vs_best")),
            "sec": h.get("duration_sec"), "reflection": short(h.get("reflection") or h.get("error_message") or "", 220),
        })
    return rows


def history_table(history: list[dict], last_n: int = 12) -> str:
    if not history:
        return "экспериментов ещё не было"
    rows = history_rows(history, last_n)
    return md_table(rows, ["exp_id", "kind", "summary", "status", "score", "delta", "sec", "reflection"],
                    ["exp", "вид", "гипотеза", "статус", "скор", "Δ к лучшему", "сек", "рефлексия/ошибка"])


def get_experiment_history(last_n: int = 3, work_dir: str | Path = "agent_work") -> str:
    """Совместимо с первой версией: JSON-строка с последними N экспериментами."""
    history = read_json(Path(work_dir) / "experiment_history.json", default=None)
    if not history:
        return json.dumps({"status": "info", "message": "История экспериментов пуста. Это первый запуск."},
                          ensure_ascii=False)
    keep = ("exp_id", "timestamp", "hypothesis_summary", "kind", "model", "status", "metric_name",
            "local_cv_score", "cv_std", "delta_vs_best", "is_new_best", "reflection", "error_message")
    recent = [{k: h.get(k) for k in keep} for h in history[-last_n:]]
    return json.dumps({"status": "success", "history": recent}, ensure_ascii=False, indent=2)


# ============================================================================ анализ ошибок

def _segments(X: pd.DataFrame, cat_cols: list[str], err: np.ndarray, kind: str, max_cols: int = 3) -> list[dict]:
    """Худшие сегменты категориальных признаков по ошибке (только достаточно крупные)."""
    out = []
    cols = [c for c in cat_cols if 2 <= X[c].nunique() <= 60][:max_cols]
    base = float(np.mean(err)) if len(err) else 0.0
    for c in cols:
        df = pd.DataFrame({"v": X[c].astype(str).values, "e": err})
        g = df.groupby("v")["e"].agg(["mean", "count"])
        g = g[g["count"] >= max(30, int(0.005 * len(df)))]
        if g.empty:
            continue
        g["contrib"] = g["mean"] * g["count"]
        g = g.sort_values("contrib", ascending=False).head(3)
        for v, r in g.iterrows():
            out.append({"feature": c, "value": v, "rows": int(r["count"]),
                        kind: round(float(r["mean"]), 4), "vs_avg": round(float(r["mean"]) / max(base, 1e-12), 2)})
    return out


def analyze_experiment_errors(exp_id: str, ws, store, compare_to: str | None = None, X: pd.DataFrame | None = None) -> dict:
    """Разобрать ошибки эксперимента по OOF. Возвращает {"report_md": str, "facts": dict}.

    compare_to — с каким экспериментом сравнивать (по умолчанию — лучший до него).
    """
    rec = store.get(exp_id)
    if not rec or rec.get("status") != "success":
        msg = rec.get("error_message") if rec else "нет такого эксперимента"
        return {"report_md": f"Эксперимент {exp_id} не дал предсказаний: {short(msg, 800)}", "facts": {}}
    oof = np.load(rec["oof_path"])
    mask = ~np.isnan(oof).any(axis=1) if oof.ndim == 2 else ~np.isnan(oof)
    y = ws.y[mask]
    p = oof[mask]
    if X is None:
        X = pd.read_pickle(ws.work_dir / "cache" / "data.pkl")["X"]
    Xm = X[mask].reset_index(drop=True)
    cat_cols = [c for c in ws.meta["cat_features"] if c in Xm.columns]
    ref = store.get(compare_to) if compare_to else None
    if ref is None:
        prev = [h for h in store.successes() if h["exp_id"] != exp_id and h["exp_id"] < exp_id]
        if prev:
            ref = max(prev, key=lambda h: h["local_cv_score"]) if store.metric.greater_is_better else \
                min(prev, key=lambda h: h["local_cv_score"])
    ref_p = None
    if ref is not None and ref.get("oof_path"):
        r = np.load(ref["oof_path"])
        r_mask = ~np.isnan(r).any(axis=1) if r.ndim == 2 else ~np.isnan(r)
        if np.array_equal(r_mask, mask):
            ref_p = r[mask]
    lines = [f"### Анализ ошибок {exp_id} ({rec.get('kind')}:{rec.get('model') or ''})",
             f"- {ws.metric.name} по OOF: **{fmt_num(rec['local_cv_score'])}**, по фолдам: "
             f"{[fmt_num(s, 4) for s in rec.get('fold_scores') or []]}"]
    if ref is not None:
        lines.append(f"- сравнение с {ref['exp_id']} ({fmt_num(ref['local_cv_score'])}): "
                     f"Δ = {fmt_num(ws.metric.delta(rec['local_cv_score'], ref['local_cv_score']))} (плюс = лучше)")
    facts: dict = {"score": rec["local_cv_score"], "compare_to": ref["exp_id"] if ref else None}

    if ws.cfg.is_classification:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        classes = [str(c) for c in ws.meta["classes"]]
        k = len(classes)
        pred = p.argmax(axis=1)
        pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
        f1_ref = None
        if ref_p is not None:
            _, _, f1_ref, _ = precision_recall_fscore_support(y, ref_p.argmax(axis=1), labels=list(range(k)), zero_division=0)
        rows = []
        for i in np.argsort(f1):
            row = {"class": classes[i], "support": int(sup[i]), "precision": float(pr[i]), "recall": float(rc[i]),
                   "f1": float(f1[i])}
            if f1_ref is not None:
                row["d_f1"] = float(f1[i] - f1_ref[i])
            rows.append(row)
        cols = ["class", "support", "precision", "recall", "f1"] + (["d_f1"] if f1_ref is not None else [])
        lines.append(f"- доля ошибок: {float((pred != y).mean()):.2%}")
        lines.append("\n**Классы (от худшего F1):**\n" + md_table(rows[:12], cols))
        cm = confusion_matrix(y, pred, labels=list(range(k)))
        np.fill_diagonal(cm, 0)
        pairs = []
        for i, j in zip(*np.unravel_index(np.argsort(cm, axis=None)[::-1], cm.shape)):
            if cm[i, j] == 0 or len(pairs) >= 6:
                break
            pairs.append({"true": classes[i], "pred": classes[j], "count": int(cm[i, j]),
                          "share_of_true": round(float(cm[i, j] / max(sup[i], 1)), 3)})
        if pairs:
            lines.append("\n**Главные путаницы (истина → предсказание):**\n" +
                         md_table(pairs, ["true", "pred", "count", "share_of_true"]))
        err = (pred != y).astype(float)
        seg = _segments(Xm, cat_cols, err, "error_rate")
        if seg:
            lines.append("\n**Сегменты с наибольшим вкладом в ошибки:**\n" +
                         md_table(seg, ["feature", "value", "rows", "error_rate", "vs_avg"]))
        conf = p.max(axis=1)
        lines.append(f"- средняя уверенность: верные {conf[pred == y].mean():.3f}, ошибочные "
                     f"{conf[pred != y].mean() if (pred != y).any() else float('nan'):.3f}")
        facts.update(worst_classes=[r["class"] for r in rows[:3]], confusions=pairs[:3], segments=seg[:3])
    else:
        res = y - p
        sq = res ** 2
        lines.append(f"- RMSE {np.sqrt(sq.mean()):.4g}, MAE {np.abs(res).mean():.4g}, смещение (y - pred) "
                     f"{res.mean():+.4g}")
        top = np.sort(sq)[::-1]
        share1 = float(top[: max(1, len(top) // 100)].sum() / max(sq.sum(), 1e-12))
        lines.append(f"- худший 1% строк даёт {share1:.1%} суммарной квадратичной ошибки")
        try:
            dec = pd.qcut(y, 10, duplicates="drop")
            g = pd.DataFrame({"d": dec, "res": res, "sq": sq}).groupby("d", observed=True).agg(
                bias=("res", "mean"), rmse=("sq", lambda s: float(np.sqrt(s.mean()))), n=("res", "size"))
            rows = [{"target_bin": f"{i.left:.4g}…{i.right:.4g}", "n": int(r["n"]), "bias": float(r["bias"]),
                     "rmse": float(r["rmse"])} for i, r in g.iterrows()]
            lines.append("\n**Ошибка по децилям таргета (bias > 0 — недооценка):**\n" +
                         md_table(rows, ["target_bin", "n", "bias", "rmse"]))
        except Exception:  # noqa: BLE001
            pass
        seg = _segments(Xm, cat_cols, sq, "mse")
        if seg:
            lines.append("\n**Сегменты с наибольшим вкладом в квадратичную ошибку:**\n" +
                         md_table(seg, ["feature", "value", "rows", "mse", "vs_avg"]))
        if ref_p is not None:
            gain = (y - ref_p) ** 2 - sq
            lines.append(f"- относительно {ref['exp_id']}: улучшились {float((gain > 0).mean()):.1%} строк, "
                         f"ухудшились {float((gain < 0).mean()):.1%}")
        facts.update(bias=float(res.mean()), worst1pct_share=share1, segments=seg[:3])
    report = "\n".join(lines)
    out_path = Path(rec["oof_path"]).parent / "analysis.md"
    out_path.write_text(report, encoding="utf-8")
    store.update(exp_id, analysis_path=str(out_path))
    return {"report_md": report, "facts": facts}


# ============================================================================ рефлексия

def rule_based_reflection(rec: dict, analysis: dict, metric) -> str:
    st = rec.get("status")
    if st != "success":
        return (f"Эксперимент не завершился ({st}): {short(rec.get('error_message'), 300)}. "
                f"Нужно исправить реализацию или выбрать более быструю конфигурацию.")
    parts = []
    d = rec.get("delta_vs_best")
    if rec.get("is_new_best"):
        parts.append(f"Новый лучший результат {metric.name}={fmt_num(rec['local_cv_score'])} (Δ={fmt_num(d)}"
                     + (", в пределах шума между фолдами — эффект не доказан" if rec.get("within_noise") else "") + ").")
    else:
        parts.append(f"Метрика {fmt_num(rec['local_cv_score'])} не лучше текущего лучшего (Δ={fmt_num(d)}), "
                     f"изменение не принимаем.")
    f = analysis.get("facts", {})
    if f.get("worst_classes"):
        parts.append(f"Хуже всего классы: {', '.join(f['worst_classes'])}.")
    if f.get("confusions"):
        c = f["confusions"][0]
        parts.append(f"Главная путаница: {c['true']} → {c['pred']} ({c['count']} строк, {c['share_of_true']:.0%} класса).")
    if "bias" in f:
        parts.append(f"Смещение {f['bias']:+.3g}; худший 1% строк даёт {f['worst1pct_share']:.0%} квадратичной ошибки.")
    if f.get("segments"):
        s = f["segments"][0]
        parts.append(f"Слабый сегмент: {s['feature']}={s['value']} (ошибка в {s['vs_avg']}× выше средней).")
    if rec.get("cv_std") and rec.get("local_cv_score"):
        parts.append(f"Разброс по фолдам {fmt_num(rec['cv_std'])}.")
    return " ".join(parts)


def write_reflection(rec: dict, analysis: dict, ws, store, llm=None) -> str:
    """Рефлексия по эксперименту: LLM, если доступна, иначе по правилам. Сохраняется в историю."""
    text = None
    # LLM-рефлексия только там, где есть что осмыслить: модели и код. Бленды, постобработка и таймауты
    # описываются правилами — экономим вызовы и токены.
    worth_llm = rec.get("kind") in ("registry", "code") and rec.get("status") in ("success", "error")
    if worth_llm and llm is not None and llm.available(4000):
        hyp = rec.get("hypothesis") or {}
        user = prompts.REFLECTION_USER.format(
            hypothesis=short(json.dumps({k: hyp.get(k) for k in ("statement", "mechanism", "expected_effect",
                                                                    "falsification", "implementation")},
                                        ensure_ascii=False, default=str), 1500),
            status=rec.get("status"), metric=ws.metric.name, score=fmt_num(rec.get("local_cv_score")),
            prev_best=fmt_num(rec.get("best_before")), delta=fmt_num(rec.get("delta_vs_best")),
            direction=ws.metric.direction,
            error=f"ошибка: {short(rec.get('error_message'), 600)}" if rec.get("status") != "success" else "",
            analysis=short(analysis.get("report_md", ""), 3500),
        )
        text = llm.chat([{"role": "system", "content": prompts.REFLECTION_SYSTEM},
                         {"role": "user", "content": user}], max_tokens=500, temperature=0.3, stage="reflection")
    if not text:
        text = rule_based_reflection(rec, analysis, ws.metric)
    text = short(text, 1200)
    store.update(rec["exp_id"], reflection=text)
    return text
