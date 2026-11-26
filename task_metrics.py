"""task_metrics.py — реестр метрик без хардкода под конкретную задачу.

Метрика задаётся строкой из task.md / task_config («Macro F1», «RMSE», «ROC-AUC», «quadratic weighted
kappa», «precision_weighted», ...). resolve_metric() нормализует имя и возвращает объект Metric:
    name       — каноническое имя
    direction  — "maximize" | "minimize" (никаких списков is_lower_better)
    needs      — что нужно на вход: "value" (регрессия), "label" (метки), "proba" (вероятности)
    fn(y_true, pred, ctx) -> float

Соглашение о предсказаниях внутри агента:
    регрессия      — pred: (n,) float
    классификация  — pred: (n, K) вероятности по классам в порядке ctx["classes"]; y_true: коды 0..K-1.
Метрики по меткам берут argmax, по вероятностям — матрицу целиком.

Если имени нет в собственном реестре, пробуем sklearn.metrics.get_scorer(name) — так поддерживается
любая метрика sklearn (направление берётся из знака скорера: neg_* — минимизация).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm

EPS = 1e-15


@dataclass
class Metric:
    name: str
    direction: str  # "maximize" | "minimize"
    needs: str      # "value" | "label" | "proba"
    fn: Callable[[np.ndarray, np.ndarray, dict], float]
    task: str = "any"  # "regression" | "classification" | "binary" | "any"

    @property
    def greater_is_better(self) -> bool:
        return self.direction == "maximize"

    def worst(self) -> float:
        return -np.inf if self.greater_is_better else np.inf

    def is_better(self, a: float | None, b: float | None) -> bool:
        """a лучше b?"""
        if a is None or (isinstance(a, float) and np.isnan(a)):
            return False
        if b is None or (isinstance(b, float) and np.isnan(b)):
            return True
        return a > b if self.greater_is_better else a < b

    def delta(self, new: float, best: float | None) -> float:
        """Положительная дельта = улучшение, в любом направлении метрики."""
        if best is None:
            return 0.0
        return (new - best) if self.greater_is_better else (best - new)


# ----------------------------------------------------------------------------- helpers

def _labels(pred: np.ndarray) -> np.ndarray:
    return pred.argmax(axis=1) if pred.ndim == 2 else np.asarray(pred).astype(int)


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def _pos(ctx: dict) -> int:
    return int(ctx.get("positive_index", 1))


def _k(ctx: dict, pred: np.ndarray) -> int:
    return int(ctx.get("n_classes") or (pred.shape[1] if pred.ndim == 2 else 2))


def _roc_auc(y, p, ctx):
    k = _k(ctx, p)
    if k == 2:
        return skm.roc_auc_score(y == _pos(ctx), p[:, _pos(ctx)])
    present = np.unique(y)
    pn = _norm(p)
    return skm.roc_auc_score(y, pn[:, present] / pn[:, present].sum(axis=1, keepdims=True),
                             multi_class="ovr", average="macro", labels=present)


def _roc_auc_weighted(y, p, ctx):
    present = np.unique(y)
    pn = _norm(p)
    return skm.roc_auc_score(y, pn[:, present] / pn[:, present].sum(axis=1, keepdims=True),
                             multi_class="ovr", average="weighted", labels=present)


def _f1_bin(y, p, ctx):
    return skm.f1_score(y == _pos(ctx), _labels(p) == _pos(ctx), zero_division=0)


def _prec_bin(y, p, ctx):
    return skm.precision_score(y == _pos(ctx), _labels(p) == _pos(ctx), zero_division=0)


def _rec_bin(y, p, ctx):
    return skm.recall_score(y == _pos(ctx), _labels(p) == _pos(ctx), zero_division=0)


def _rmsle(y, p, ctx):
    return float(np.sqrt(skm.mean_squared_log_error(np.clip(y, 0, None), np.clip(p, 0, None))))


def _smape(y, p, ctx):
    denom = (np.abs(y) + np.abs(p))
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(p - y) / denom) * 100)


def _wape(y, p, ctx):
    return float(np.sum(np.abs(y - p)) / max(np.sum(np.abs(y)), EPS))


_R = "regression"
_C = "classification"
_B = "binary"

_REGISTRY: dict[str, Metric] = {}


def _add(name, direction, needs, fn, task="any", aliases=()):
    m = Metric(name, direction, needs, fn, task)
    _REGISTRY[name] = m
    for a in (name, *aliases):
        _ALIASES[_key(a)] = name


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9а-яё]", "", str(name).lower())


_ALIASES: dict[str, str] = {}

# --- регрессия
_add("rmse", "minimize", "value", lambda y, p, c: float(np.sqrt(skm.mean_squared_error(y, p))), _R,
     ("root mean squared error", "root_mean_squared_error", "neg_root_mean_squared_error", "среднеквадратичная ошибка корень"))
_add("mse", "minimize", "value", lambda y, p, c: float(skm.mean_squared_error(y, p)), _R,
     ("mean squared error", "neg_mean_squared_error", "l2"))
_add("mae", "minimize", "value", lambda y, p, c: float(skm.mean_absolute_error(y, p)), _R,
     ("mean absolute error", "neg_mean_absolute_error", "l1", "средняя абсолютная ошибка"))
_add("medae", "minimize", "value", lambda y, p, c: float(skm.median_absolute_error(y, p)), _R,
     ("median absolute error", "neg_median_absolute_error"))
_add("rmsle", "minimize", "value", _rmsle, _R, ("root mean squared log error", "neg_root_mean_squared_log_error"))
_add("msle", "minimize", "value", lambda y, p, c: float(skm.mean_squared_log_error(np.clip(y, 0, None), np.clip(p, 0, None))), _R,
     ("mean squared log error", "neg_mean_squared_log_error"))
_add("mape", "minimize", "value", lambda y, p, c: float(skm.mean_absolute_percentage_error(y, p)), _R,
     ("mean absolute percentage error", "neg_mean_absolute_percentage_error"))
_add("smape", "minimize", "value", _smape, _R, ("symmetric mape",))
_add("wape", "minimize", "value", _wape, _R, ("wmape",))
_add("r2", "maximize", "value", lambda y, p, c: float(skm.r2_score(y, p)), _R, ("r2 score", "r^2", "coefficient of determination", "коэффициент детерминации"))

# --- классификация по меткам
_add("accuracy", "maximize", "label", lambda y, p, c: float(skm.accuracy_score(y, _labels(p))), _C, ("acc", "доля верных ответов", "точность классификации"))
_add("balanced_accuracy", "maximize", "label", lambda y, p, c: float(skm.balanced_accuracy_score(y, _labels(p))), _C, ("balanced accuracy",))
_add("f1_macro", "maximize", "label", lambda y, p, c: float(skm.f1_score(y, _labels(p), average="macro", zero_division=0)), _C,
     ("macro f1", "macro_f1", "f1 macro", "macro-f1", "macrof1", "f1-macro", "макро f1"))
_add("f1_micro", "maximize", "label", lambda y, p, c: float(skm.f1_score(y, _labels(p), average="micro", zero_division=0)), _C,
     ("micro f1", "micro_f1"))
_add("f1_weighted", "maximize", "label", lambda y, p, c: float(skm.f1_score(y, _labels(p), average="weighted", zero_division=0)), _C,
     ("weighted f1", "weighted_f1", "f1 weighted"))
_add("f1", "maximize", "label", _f1_bin, _B, ("f1 score", "f1-score", "f1_binary", "f-score", "f-мера"))
_add("precision", "maximize", "label", _prec_bin, _B, ("precision_binary",))
_add("recall", "maximize", "label", _rec_bin, _B, ("recall_binary", "sensitivity", "полнота"))
_add("precision_macro", "maximize", "label", lambda y, p, c: float(skm.precision_score(y, _labels(p), average="macro", zero_division=0)), _C, ("macro precision",))
_add("recall_macro", "maximize", "label", lambda y, p, c: float(skm.recall_score(y, _labels(p), average="macro", zero_division=0)), _C, ("macro recall",))
_add("precision_weighted", "maximize", "label", lambda y, p, c: float(skm.precision_score(y, _labels(p), average="weighted", zero_division=0)), _C, ("weighted precision",))
_add("recall_weighted", "maximize", "label", lambda y, p, c: float(skm.recall_score(y, _labels(p), average="weighted", zero_division=0)), _C, ("weighted recall",))
_add("mcc", "maximize", "label", lambda y, p, c: float(skm.matthews_corrcoef(y, _labels(p))), _C, ("matthews", "matthews corrcoef", "matthews_corrcoef"))
_add("cohen_kappa", "maximize", "label", lambda y, p, c: float(skm.cohen_kappa_score(y, _labels(p))), _C, ("kappa", "cohen kappa"))
_add("qwk", "maximize", "label", lambda y, p, c: float(skm.cohen_kappa_score(y, _labels(p), weights="quadratic")), _C,
     ("quadratic weighted kappa", "quadratic kappa", "weighted kappa"))

# --- по вероятностям
_add("roc_auc", "maximize", "proba", _roc_auc, _C, ("auc", "roc auc", "roc-auc", "rocauc", "auroc", "roc_auc_ovr", "auc roc", "площадь под roc"))
_add("roc_auc_weighted", "maximize", "proba", _roc_auc_weighted, _C, ("roc_auc_ovr_weighted", "weighted auc"))
_add("gini", "maximize", "proba", lambda y, p, c: 2 * _roc_auc(y, p, c) - 1, _C, ("gini coefficient", "джини"))
_add("logloss", "minimize", "proba", lambda y, p, c: float(skm.log_loss(y, _norm(p), labels=list(range(_k(c, p))))), _C,
     ("log loss", "log_loss", "neg_log_loss", "cross entropy", "cross-entropy", "multi logloss", "mlogloss", "binary crossentropy"))
_add("average_precision", "maximize", "proba", lambda y, p, c: float(skm.average_precision_score(y == _pos(c), p[:, _pos(c)])), _B,
     ("pr auc", "pr-auc", "prauc", "average precision", "auprc"))
_add("brier", "minimize", "proba", lambda y, p, c: float(skm.brier_score_loss(y == _pos(c), p[:, _pos(c)])), _B,
     ("brier score", "neg_brier_score"))


def _from_sklearn(name: str) -> Metric | None:
    """Обернуть любой sklearn-скорер по имени (get_scorer)."""
    try:
        scorer = skm.get_scorer(name)
    except Exception:  # noqa: BLE001
        return None
    score_func = getattr(scorer, "_score_func", None)
    if score_func is None:
        return None
    sign = getattr(scorer, "_sign", 1)
    kwargs = dict(getattr(scorer, "_kwargs", {}) or {})
    resp = getattr(scorer, "_response_method", "predict")
    resp = resp[0] if isinstance(resp, (tuple, list)) else resp
    needs = "proba" if resp in ("predict_proba", "decision_function") else "label"

    def fn(y, p, c, _f=score_func, _kw=kwargs, _needs=needs):
        if p.ndim == 1:  # регрессия
            return float(_f(y, p, **_kw))
        if _needs == "label":
            return float(_f(y, _labels(p), **_kw))
        pp = p[:, _pos(c)] if p.shape[1] == 2 else _norm(p)
        return float(_f(y, pp, **_kw))

    return Metric(name, "maximize" if sign > 0 else "minimize", needs, fn)


def resolve_metric(name: str | None, task_type: str | None = None) -> Metric | None:
    """Найти метрику по имени. task_type уточняет неоднозначности (f1/auc для мультикласса)."""
    if not name:
        return None
    key = _key(name)
    canon = _ALIASES.get(key)
    multiclass = task_type == "multiclass_classification"
    if canon is None:
        sk = _from_sklearn(str(name).strip())
        if sk is not None:
            return sk
        for alias_key, c in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if len(alias_key) >= 3 and key.startswith(alias_key):
                canon = c
                break
    if canon is not None:
        if multiclass and canon in ("f1", "precision", "recall"):
            canon = f"{canon}_macro"
        if multiclass and canon in ("average_precision", "brier"):
            canon = "roc_auc" if canon == "average_precision" else "logloss"
        return _REGISTRY[canon]
    return _from_sklearn(str(name).strip())


def default_metric(task_type: str, prediction_format: str = "label") -> Metric:
    if task_type == "regression":
        return _REGISTRY["rmse"]
    if task_type == "binary_classification":
        return _REGISTRY["roc_auc"] if prediction_format.startswith("proba") else _REGISTRY["f1"]
    return _REGISTRY["logloss"] if prediction_format == "proba_per_class" else _REGISTRY["f1_macro"]


# порядок важен: более специфичные шаблоны раньше
_TEXT_PATTERNS: list[tuple[str, str]] = [
    (r"macro[\s_-]*f1|f1[\s_-]*macro|макро[\s-]*f1", "f1_macro"),
    (r"micro[\s_-]*f1|f1[\s_-]*micro", "f1_micro"),
    (r"weighted[\s_-]*f1|f1[\s_-]*weighted|взвешенн\w*\s+f1", "f1_weighted"),
    (r"quadratic[\s_-]*weighted[\s_-]*kappa|\bqwk\b|квадратичн\w*\s+каппа", "qwk"),
    (r"cohen'?s?[\s_-]*kappa|каппа\s+коэна", "cohen_kappa"),
    (r"\brmsle\b|root\s+mean\s+squared\s+log", "rmsle"),
    (r"\bmsle\b", "msle"),
    (r"\brmse\b|root\s+mean\s+squared\s+error", "rmse"),
    (r"\bmse\b|mean\s+squared\s+error", "mse"),
    (r"\bsmape\b", "smape"),
    (r"\bw?mape\b|mean\s+absolute\s+percentage", "mape"),
    (r"\bmae\b|mean\s+absolute\s+error", "mae"),
    (r"\br2\b|r\^2|r²|коэффициент\s+детерминации", "r2"),
    (r"pr[\s_-]*auc|average\s+precision|auprc", "average_precision"),
    (r"\bgini\b|джини", "gini"),
    (r"roc[\s_-]*auc|\bauc\b|auroc", "roc_auc"),
    (r"log[\s_-]*loss|cross[\s_-]*entropy|логлосс", "logloss"),
    (r"balanced[\s_-]*accuracy", "balanced_accuracy"),
    (r"matthews|\bmcc\b", "mcc"),
    (r"\bbrier\b", "brier"),
    (r"\bf1\b|f1[\s_-]*score|f-?мера", "f1"),
    (r"\baccuracy\b|доля\s+правильн", "accuracy"),
    (r"\bprecision\b", "precision"),
    (r"\brecall\b|полнот", "recall"),
]


def detect_metric_in_text(text: str) -> str | None:
    """Эвристика: найти основную метрику в тексте условия.

    Сначала ищем в строках, где есть слова «метрик/metric/оценк/score», и берём самое раннее упоминание;
    строки со словом «основн/primary/main» в приоритете.
    """
    lines = text.splitlines()
    pri, sec = [], []
    for ln in lines:
        low = ln.lower()
        if re.search(r"метрик|metric|оценк|scor|evaluat", low):
            (pri if re.search(r"основн|primary|main|главн", low) else sec).append(low)
    for group in (pri, sec, [text.lower()]):
        best = None
        for li, ln in enumerate(group):
            for pat, name in _TEXT_PATTERNS:
                m = re.search(pat, ln)
                if m and (best is None or (li, m.start()) < best[0]):
                    best = ((li, m.start()), name)
        if best:
            return best[1]
    return None


def evaluate(metric: Metric, y_true: np.ndarray, pred: np.ndarray, ctx: dict) -> float:
    """Посчитать метрику. Для классификации pred — (n, K) вероятности."""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred, dtype=float)
    return float(metric.fn(y_true, pred, ctx))


def list_metric_names() -> list[str]:
    return sorted(_REGISTRY)
