"""model_zoo.py — реестр моделей с единым контрактом.

    val_pred, test_pred = run_model(name, params, X_train, y_train, X_val, X_test, ctx, features=None)

Контракт (тот же, что у функций fit_predict, которые пишет LLM):
    X_*      — pandas.DataFrame с одинаковыми колонками; категориальные — строки (или любые нечисловые)
    y_train  — классификация: коды 0..K-1; регрессия: float
    ctx      — {"task_type", "n_classes", "n_threads", "seed", ...}
    возврат  — регрессия: (n,) ; классификация: (n, K) вероятности в порядке кодов 0..K-1

Все модели:
  * сами кодируют категориальные признаки под себя (CatBoost — нативно, LightGBM/XGBoost — category,
    sklearn — порядковые коды или one-hot);
  * делают early stopping на ВНУТРЕННЕМ куске обучающего фолда (10%), а не на валидационном фолде —
    так валидационная метрика честная (в старом ml_experiment eval_set был тем же фолдом, что и оценка);
  * переживают отсутствие класса в обучающем куске (вероятность такого класса = 0);
  * принимают любые параметры конструктора из params (LLM может задать loss/objective и т.п.) и служебные:
        early_stopping_rounds (int), class_balance (bool), target_transform ("log1p"), es_fraction (float),
        seed_offset (int — та же модель с другим сидом, для бэггинга в бленде).
Если библиотеки нет в окружении, модель не попадает в available_models() — агент не упадёт.
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DEFAULTS: dict[str, dict] = {
    "catboost": {"iterations": 3000, "learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 3.0},
    "lightgbm": {"n_estimators": 4000, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 20,
                 "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "xgboost": {"n_estimators": 4000, "learning_rate": 0.05, "max_depth": 7, "subsample": 0.8,
                "colsample_bytree": 0.8, "min_child_weight": 1.0, "reg_lambda": 1.0, "max_bin": 256},
    "hist_gb": {"max_iter": 1500, "learning_rate": 0.06, "max_leaf_nodes": 63, "min_samples_leaf": 20,
                "l2_regularization": 0.0},
    "random_forest": {"n_estimators": 300, "min_samples_leaf": 2, "max_features": "sqrt"},
    "extra_trees": {"n_estimators": 400, "min_samples_leaf": 2, "max_features": "sqrt"},
    "linear": {},
}

# быстрые параметры для первого бейзлайна (минуты, а не десятки минут)
FAST: dict[str, dict] = {
    "catboost": {"iterations": 500, "learning_rate": 0.12, "depth": 6},
    "lightgbm": {"n_estimators": 600, "learning_rate": 0.1, "num_leaves": 63},
    "xgboost": {"n_estimators": 600, "learning_rate": 0.1, "max_depth": 7},
    "hist_gb": {"max_iter": 300, "learning_rate": 0.12, "max_leaf_nodes": 63},
    "random_forest": {"n_estimators": 150},
    "extra_trees": {"n_estimators": 200},
    "linear": {},
}

DESCRIPTIONS = {
    "catboost": "CatBoost (нативные категории, сильный дефолт; loss_function, depth, l2_leaf_reg, auto_class_weights ...)",
    "lightgbm": "LightGBM (быстрый бустинг; objective, num_leaves, min_child_samples, reg_alpha/lambda ...)",
    "xgboost": "XGBoost hist (enable_categorical; objective, max_depth, min_child_weight ...)",
    "hist_gb": "sklearn HistGradientBoosting (всегда есть; loss, max_leaf_nodes, l2_regularization ...)",
    "random_forest": "sklearn RandomForest (n_estimators, max_depth, min_samples_leaf, max_features)",
    "extra_trees": "sklearn ExtraTrees (разнообразие для ансамбля)",
    "linear": "линейная модель: LogisticRegression / Ridge с one-hot и масштабированием (C или alpha)",
}

_LIB = {"catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost"}

SPECIAL_KEYS = ("early_stopping_rounds", "class_balance", "target_transform", "es_fraction", "seed_offset")


def available_models() -> dict[str, str]:
    out = {}
    for name, desc in DESCRIPTIONS.items():
        lib = _LIB.get(name)
        if lib is None or importlib.util.find_spec(lib) is not None:
            out[name] = desc
    return out


def preferred_gbdt_order() -> list[str]:
    av = available_models()
    return [m for m in ("catboost", "lightgbm", "xgboost", "hist_gb") if m in av]


def work_units(model: str | None, params: dict | None = None, features: dict | None = None) -> float:
    """Относительный объём работы конфигурации (1.0 ≈ быстрый бейзлайн этой модели).

    Нужен ResourceController: время на единицу работы меряется по факту, а новая конфигурация
    той же модели (больше итераций, меньше шаг, глубже) получает пропорционально больший таймаут.
    Число раундов бустинга с early stopping грубо оценивается как min(лимит, 30 / learning_rate);
    для CatBoost (симметричные деревья) стоимость растёт линейно с глубиной.
    """
    name = normalize_model_name(model) or ""
    p = {**DEFAULTS.get(name, {}), **(params or {})}
    base = {**DEFAULTS.get(name, {}), **FAST.get(name, {})}

    def rounds(q, it_key):
        lr = float(q.get("learning_rate", q.get("eta", 0.1)) or 0.1)
        return min(float(q.get(it_key, q.get("iterations", q.get("n_estimators", 1000))) or 1000), 30.0 / max(lr, 1e-3))

    try:
        if name == "catboost":
            u = rounds(p, "iterations") * float(p.get("depth", 6)) / (rounds(base, "iterations") * float(base.get("depth", 6)))
        elif name == "lightgbm":
            u = rounds(p, "n_estimators") * float(p.get("num_leaves", 63)) / (rounds(base, "n_estimators") * float(base.get("num_leaves", 63)))
        elif name == "xgboost":
            u = rounds(p, "n_estimators") * 2 ** ((float(p.get("max_depth", 7)) - 7) / 2) / \
                (rounds(base, "n_estimators") * 2 ** ((float(base.get("max_depth", 7)) - 7) / 2))
        elif name == "hist_gb":
            u = rounds(p, "max_iter") * float(p.get("max_leaf_nodes", 63) or 63) / (rounds(base, "max_iter") * float(base.get("max_leaf_nodes", 63) or 63))
        elif name in ("random_forest", "extra_trees"):
            u = float(p.get("n_estimators", 300)) / float(base.get("n_estimators", 300))
        else:
            u = 1.0
    except (TypeError, ValueError):
        u = 1.0
    if (features or {}).get("frequency_encoding"):
        u *= 1.1
    return float(min(max(u, 0.1), 50.0))


# ----------------------------------------------------------------------------- helpers

def _is_clf(ctx: dict) -> bool:
    return ctx.get("task_type", "regression") != "regression"


def _cat_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or isinstance(X[c].dtype, pd.CategoricalDtype)]


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "__nan__").map(str)


def _union_categories(frames: list[pd.DataFrame], col: str) -> pd.Index:
    return pd.Index(pd.unique(pd.concat([_as_str(f[col]) for f in frames], ignore_index=True)))


def _to_category(frames: list[pd.DataFrame], cat_cols: list[str]) -> list[pd.DataFrame]:
    out = [f.copy() for f in frames]
    for c in cat_cols:
        cats = _union_categories(frames, c)
        for f in out:
            f[c] = pd.Categorical(_as_str(f[c]), categories=cats)
    return out


def _to_codes(frames: list[pd.DataFrame], cat_cols: list[str]) -> tuple[list[np.ndarray], list[bool]]:
    """Порядковые коды категорий (общий словарь по всем кадрам) -> float-матрицы."""
    out = [f.copy() for f in frames]
    is_cat = []
    for c in out[0].columns:
        if c in cat_cols:
            cats = _union_categories(frames, c)
            for f in out:
                f[c] = pd.Categorical(_as_str(f[c]), categories=cats).codes.astype(float)
            is_cat.append(len(cats) < 250)
        else:
            for f in out:
                f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)
            is_cat.append(False)
    return [f.values.astype(np.float32) for f in out], is_cat


def _inner_split(y: np.ndarray, clf: bool, frac: float, seed: int):
    n = len(y)
    if n < 300 or frac <= 0:
        return np.arange(n), None
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

    if clf and len(np.unique(y)) > 1:
        counts = np.bincount(y)
        if counts[counts > 0].min() >= 2:
            sp = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
            return next(sp.split(np.zeros(n), y))
    sp = ShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    return next(sp.split(np.zeros(n)))


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y)
    w = len(y) / (np.maximum(counts, 1) * max(1, (counts > 0).sum()))
    return w[y]


def _local_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    present = np.unique(y)
    remap = {c: i for i, c in enumerate(present)}
    return present, np.array([remap[v] for v in y], dtype=int)


def _align(proba: np.ndarray, present: np.ndarray, k: int) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        proba = np.c_[1 - proba, proba]
    out = np.zeros((proba.shape[0], k))
    out[:, present] = proba[:, : len(present)]
    s = out.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return out / s


def _const_pred(n_val: int, n_test: int, present: np.ndarray, k: int):
    p = np.zeros(k)
    p[present[0]] = 1.0
    return np.tile(p, (n_val, 1)), np.tile(p, (n_test, 1))


def _split_params(name: str, params: dict) -> tuple[dict, dict]:
    p = {**DEFAULTS.get(name, {}), **(params or {})}
    special = {k: p.pop(k) for k in SPECIAL_KEYS if k in p}
    return p, special


# ----------------------------------------------------------------------------- models

def _catboost(params, X_tr, y_tr, X_va, X_te, ctx):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    clf = _is_clf(ctx)
    p, sp = _split_params("catboost", params)
    es_rounds = int(sp.get("early_stopping_rounds", p.pop("od_wait", 150)))
    for k in ("cat_features", "od_type", "use_best_model", "eval_set", "train_dir"):
        p.pop(k, None)  # харнесс задаёт сам; иначе CatBoost ругается на конфликт параметров
    if "num_boost_round" in p or "n_estimators" in p:
        p["iterations"] = int(p.pop("num_boost_round", p.pop("n_estimators", p.get("iterations", 3000))))
    p.setdefault("random_seed", ctx.get("seed", 42))
    p.setdefault("thread_count", ctx.get("n_threads", -1))
    p.setdefault("verbose", 0)
    p.setdefault("allow_writing_files", False)
    cats = _cat_cols(X_tr)
    frames = [X_tr.copy(), X_va.copy(), X_te.copy()]
    for f in frames:
        for c in cats:
            f[c] = _as_str(f[c])
    X_tr, X_va, X_te = frames
    cat_idx = [X_tr.columns.get_loc(c) for c in cats]
    if clf:
        k = int(ctx["n_classes"])
        present, y_loc = _local_labels(y_tr)
        if len(present) == 1:
            return _const_pred(len(X_va), len(X_te), present, k)
        p.setdefault("loss_function", "MultiClass" if len(present) > 2 else "Logloss")
        if sp.get("class_balance") and "class_weights" not in p:
            p.setdefault("auto_class_weights", "Balanced")
        model = CatBoostClassifier(**p)
    else:
        y_loc = y_tr
        model = CatBoostRegressor(**p)
    fit_idx, es_idx = _inner_split(y_loc if clf else np.zeros(len(y_loc), int), clf, float(sp.get("es_fraction", 0.1)),
                                   ctx.get("seed", 42))
    if es_idx is not None and es_rounds > 0:
        model.fit(Pool(X_tr.iloc[fit_idx], y_loc[fit_idx], cat_features=cat_idx),
                  eval_set=Pool(X_tr.iloc[es_idx], y_loc[es_idx], cat_features=cat_idx),
                  early_stopping_rounds=es_rounds, use_best_model=True)
    else:
        model.fit(Pool(X_tr, y_loc, cat_features=cat_idx))
    pv, pt = Pool(X_va, cat_features=cat_idx), Pool(X_te, cat_features=cat_idx)
    if clf:
        order = np.asarray(model.classes_).astype(int)
        return (_align(model.predict_proba(pv), present[order], k),
                _align(model.predict_proba(pt), present[order], k))
    return np.asarray(model.predict(pv), float), np.asarray(model.predict(pt), float)


def _lightgbm(params, X_tr, y_tr, X_va, X_te, ctx):
    import lightgbm as lgb

    clf = _is_clf(ctx)
    p, sp = _split_params("lightgbm", params)
    es_rounds = int(sp.get("early_stopping_rounds", 150))
    p.setdefault("random_state", ctx.get("seed", 42))
    p.setdefault("n_jobs", ctx.get("n_threads", -1))
    p.setdefault("verbose", -1)
    X_tr, X_va, X_te = _to_category([X_tr, X_va, X_te], _cat_cols(X_tr))
    if clf:
        k = int(ctx["n_classes"])
        present, y_loc = _local_labels(y_tr)
        if len(present) == 1:
            return _const_pred(len(X_va), len(X_te), present, k)
        if sp.get("class_balance"):
            p.setdefault("class_weight", "balanced")
        model = lgb.LGBMClassifier(**p)
    else:
        y_loc = y_tr
        model = lgb.LGBMRegressor(**p)
    fit_idx, es_idx = _inner_split(y_loc if clf else np.zeros(len(y_loc), int), clf, float(sp.get("es_fraction", 0.1)),
                                   ctx.get("seed", 42))
    if es_idx is not None and es_rounds > 0:
        model.fit(X_tr.iloc[fit_idx], y_loc[fit_idx], eval_set=[(X_tr.iloc[es_idx], y_loc[es_idx])],
                  callbacks=[lgb.early_stopping(es_rounds, verbose=False), lgb.log_evaluation(0)])
    else:
        model.fit(X_tr, y_loc)
    if clf:
        order = np.asarray(model.classes_).astype(int)
        return _align(model.predict_proba(X_va), present[order], k), _align(model.predict_proba(X_te), present[order], k)
    return np.asarray(model.predict(X_va), float), np.asarray(model.predict(X_te), float)


def _xgboost(params, X_tr, y_tr, X_va, X_te, ctx):
    import xgboost as xgb

    clf = _is_clf(ctx)
    p, sp = _split_params("xgboost", params)
    es_rounds = int(sp.get("early_stopping_rounds", 150))
    p.setdefault("random_state", ctx.get("seed", 42))
    p.setdefault("n_jobs", ctx.get("n_threads", -1))
    p.setdefault("tree_method", "hist")
    p.setdefault("enable_categorical", True)
    X_tr, X_va, X_te = _to_category([X_tr, X_va, X_te], _cat_cols(X_tr))
    weights = None
    if clf:
        k = int(ctx["n_classes"])
        present, y_loc = _local_labels(y_tr)
        if len(present) == 1:
            return _const_pred(len(X_va), len(X_te), present, k)
        if sp.get("class_balance"):
            weights = balanced_weights(y_loc)
        cls = xgb.XGBClassifier
    else:
        y_loc = y_tr
        cls = xgb.XGBRegressor
    fit_idx, es_idx = _inner_split(y_loc if clf else np.zeros(len(y_loc), int), clf, float(sp.get("es_fraction", 0.1)),
                                   ctx.get("seed", 42))
    if es_idx is not None and es_rounds > 0:
        p["early_stopping_rounds"] = es_rounds
        model = cls(**p)
        model.fit(X_tr.iloc[fit_idx], y_loc[fit_idx],
                  sample_weight=None if weights is None else weights[fit_idx],
                  eval_set=[(X_tr.iloc[es_idx], y_loc[es_idx])], verbose=False)
    else:
        p.pop("early_stopping_rounds", None)
        model = cls(**p)
        model.fit(X_tr, y_loc, sample_weight=weights, verbose=False)
    if clf:
        return (_align(model.predict_proba(X_va), present, k), _align(model.predict_proba(X_te), present, k))
    return np.asarray(model.predict(X_va), float), np.asarray(model.predict(X_te), float)


def _hist_gb(params, X_tr, y_tr, X_va, X_te, ctx):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    clf = _is_clf(ctx)
    p, sp = _split_params("hist_gb", params)
    (A_tr, A_va, A_te), is_cat = _to_codes([X_tr, X_va, X_te], _cat_cols(X_tr))
    p.setdefault("random_state", ctx.get("seed", 42))
    p.setdefault("early_stopping", len(A_tr) >= 1000)
    p.setdefault("validation_fraction", float(sp.get("es_fraction", 0.1)))
    p.setdefault("n_iter_no_change", int(sp.get("early_stopping_rounds", 50)))
    p["categorical_features"] = np.array(is_cat) if any(is_cat) else None
    if clf:
        k = int(ctx["n_classes"])
        present, y_loc = _local_labels(y_tr)
        if len(present) == 1:
            return _const_pred(len(A_va), len(A_te), present, k)
        if sp.get("class_balance"):
            p.setdefault("class_weight", "balanced")
        model = HistGradientBoostingClassifier(**p).fit(A_tr, y_loc)
        return _align(model.predict_proba(A_va), present, k), _align(model.predict_proba(A_te), present, k)
    model = HistGradientBoostingRegressor(**p).fit(A_tr, y_tr)
    return model.predict(A_va), model.predict(A_te)


def _forest(kind):
    def fit(params, X_tr, y_tr, X_va, X_te, ctx):
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor

        clf = _is_clf(ctx)
        p, sp = _split_params(kind, params)
        (A_tr, A_va, A_te), _ = _to_codes([X_tr, X_va, X_te], _cat_cols(X_tr))
        A_tr, A_va, A_te = (np.nan_to_num(a, nan=-999.0) for a in (A_tr, A_va, A_te))
        p.setdefault("random_state", ctx.get("seed", 42))
        p.setdefault("n_jobs", ctx.get("n_threads", -1))
        if clf:
            k = int(ctx["n_classes"])
            present, y_loc = _local_labels(y_tr)
            if len(present) == 1:
                return _const_pred(len(A_va), len(A_te), present, k)
            if sp.get("class_balance"):
                p.setdefault("class_weight", "balanced_subsample")
            cls = RandomForestClassifier if kind == "random_forest" else ExtraTreesClassifier
            model = cls(**p).fit(A_tr, y_loc)
            return _align(model.predict_proba(A_va), present, k), _align(model.predict_proba(A_te), present, k)
        cls = RandomForestRegressor if kind == "random_forest" else ExtraTreesRegressor
        p.pop("class_weight", None)
        model = cls(**p).fit(A_tr, y_tr)
        return model.predict(A_va), model.predict(A_te)
    return fit


def _linear(params, X_tr, y_tr, X_va, X_te, ctx):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    clf = _is_clf(ctx)
    p, sp = _split_params("linear", params)
    cats = _cat_cols(X_tr)
    nums = [c for c in X_tr.columns if c not in cats]
    frames = []
    for f in (X_tr, X_va, X_te):
        f = f.copy()
        for c in cats:
            f[c] = _as_str(f[c])
        for c in nums:
            f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)
        frames.append(f)
    X_tr, X_va, X_te = frames
    signed_log = FunctionTransformer(lambda a: np.sign(a) * np.log1p(np.abs(a)), feature_names_out="one-to-one")
    transformers = []
    if nums:
        transformers.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), signed_log,
                                                  StandardScaler()), nums))
    if cats:
        transformers.append(("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=50,
                                                  min_frequency=5), cats))
    pre = ColumnTransformer(transformers, sparse_threshold=0.3)
    if clf:
        k = int(ctx["n_classes"])
        present, y_loc = _local_labels(y_tr)
        if len(present) == 1:
            return _const_pred(len(X_va), len(X_te), present, k)
        if "alpha" in p and "C" not in p:
            p["C"] = 1.0 / max(float(p.pop("alpha")), 1e-6)
        p.pop("alpha", None)
        p.setdefault("max_iter", 400)
        p.setdefault("C", 1.0)
        if sp.get("class_balance"):
            p.setdefault("class_weight", "balanced")
        model = make_pipeline(pre, LogisticRegression(**p)).fit(X_tr, y_loc)
        return _align(model.predict_proba(X_va), present, k), _align(model.predict_proba(X_te), present, k)
    if "C" in p and "alpha" not in p:  # LLM может прислать параметры логрегрессии — переводим
        p["alpha"] = 1.0 / max(float(p["C"]), 1e-6)
    p = {k: v for k, v in p.items() if k in ("alpha", "fit_intercept", "solver", "max_iter", "tol", "positive")}
    p.setdefault("alpha", 1.0)
    model = make_pipeline(pre, Ridge(**p)).fit(X_tr, y_tr)
    return model.predict(X_va), model.predict(X_te)


_IMPL = {
    "catboost": _catboost, "lightgbm": _lightgbm, "xgboost": _xgboost, "hist_gb": _hist_gb,
    "random_forest": _forest("random_forest"), "extra_trees": _forest("extra_trees"), "linear": _linear,
}

_NAME_ALIASES = {
    "catboostclassifier": "catboost", "catboostregressor": "catboost", "cb": "catboost",
    "lgbm": "lightgbm", "lgbmclassifier": "lightgbm", "lgbmregressor": "lightgbm", "lgb": "lightgbm",
    "xgb": "xgboost", "xgbclassifier": "xgboost", "xgbregressor": "xgboost",
    "histgradientboosting": "hist_gb", "histgradientboostingclassifier": "hist_gb",
    "histgradientboostingregressor": "hist_gb", "hgb": "hist_gb", "histgb": "hist_gb",
    "randomforest": "random_forest", "randomforestclassifier": "random_forest", "randomforestregressor": "random_forest",
    "rf": "random_forest", "extratrees": "extra_trees", "extratreesclassifier": "extra_trees",
    "extratreesregressor": "extra_trees", "et": "extra_trees",
    "logisticregression": "linear", "logreg": "linear", "ridge": "linear", "linear": "linear", "linearmodel": "linear",
}


def normalize_model_name(name: str | None) -> str | None:
    if not name:
        return None
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if key in _IMPL:
        return key
    for k in _IMPL:
        if key == k.replace("_", ""):
            return k
    if key in _NAME_ALIASES:
        return _NAME_ALIASES[key]
    for alias, canon in _NAME_ALIASES.items():
        if alias in key:
            return canon
    return None


# ----------------------------------------------------------------------------- tuning

TUNABLE = ("catboost", "lightgbm", "xgboost", "hist_gb")


def suggest_params(name: str, trial, base: dict | None = None) -> dict:
    """Пространство поиска для тюнинга (trial — optuna.Trial или совместимый объект).

    Особые ключи (class_balance, target_transform, ...) и число итераций берутся из base:
    early stopping сам подбирает число деревьев.
    """
    base = dict(base or {})
    keep = {k: v for k, v in base.items() if k in SPECIAL_KEYS}
    f, i = trial.suggest_float, trial.suggest_int
    if name == "catboost":
        p = {"iterations": int(base.get("iterations", 3000)), "learning_rate": f("learning_rate", 0.02, 0.15, log=True),
             "depth": i("depth", 4, 9), "l2_leaf_reg": f("l2_leaf_reg", 0.5, 30.0, log=True),
             "random_strength": f("random_strength", 0.0, 3.0)}
    elif name == "lightgbm":
        p = {"n_estimators": int(base.get("n_estimators", 4000)), "learning_rate": f("learning_rate", 0.015, 0.12, log=True),
             "num_leaves": i("num_leaves", 15, 255, log=True), "min_child_samples": i("min_child_samples", 5, 150, log=True),
             "subsample": f("subsample", 0.5, 1.0), "subsample_freq": 1, "colsample_bytree": f("colsample_bytree", 0.4, 1.0),
             "reg_lambda": f("reg_lambda", 1e-3, 30.0, log=True), "reg_alpha": f("reg_alpha", 1e-3, 10.0, log=True)}
    elif name == "xgboost":
        p = {"n_estimators": int(base.get("n_estimators", 4000)), "learning_rate": f("learning_rate", 0.015, 0.12, log=True),
             "max_depth": i("max_depth", 3, 10), "min_child_weight": f("min_child_weight", 0.3, 30.0, log=True),
             "subsample": f("subsample", 0.5, 1.0), "colsample_bytree": f("colsample_bytree", 0.4, 1.0),
             "reg_lambda": f("reg_lambda", 1e-3, 30.0, log=True), "max_bin": 256}
    elif name == "hist_gb":
        p = {"max_iter": int(base.get("max_iter", 1500)), "learning_rate": f("learning_rate", 0.02, 0.15, log=True),
             "max_leaf_nodes": i("max_leaf_nodes", 15, 255, log=True), "min_samples_leaf": i("min_samples_leaf", 5, 150, log=True),
             "l2_regularization": f("l2_regularization", 1e-4, 30.0, log=True)}
    else:
        raise ValueError(f"тюнинг для {name} не поддерживается")
    p.update(keep)
    return p


class RandomTrial:
    """Замена optuna.Trial, если optuna недоступна: случайный поиск с тем же API."""

    def __init__(self, rng: np.random.Generator):
        self.rng, self.params = rng, {}

    def suggest_float(self, name, low, high, log=False):
        v = float(np.exp(self.rng.uniform(np.log(low), np.log(high)))) if log else float(self.rng.uniform(low, high))
        self.params[name] = v
        return v

    def suggest_int(self, name, low, high, log=False):
        v = int(round(np.exp(self.rng.uniform(np.log(low), np.log(high))))) if log else int(self.rng.integers(low, high + 1))
        v = int(min(max(v, low), high))
        self.params[name] = v
        return v


# ----------------------------------------------------------------------------- feature options

def apply_feature_options(X_tr, X_va, X_te, features: dict | None):
    if not features:
        return X_tr, X_va, X_te
    X_tr, X_va, X_te = X_tr.copy(), X_va.copy(), X_te.copy()
    drop = [c for c in features.get("drop_columns", []) or [] if c in X_tr.columns]
    if drop:
        X_tr, X_va, X_te = X_tr.drop(columns=drop), X_va.drop(columns=drop), X_te.drop(columns=drop)
    if features.get("frequency_encoding"):
        for c in _cat_cols(X_tr):
            all_vals = pd.concat([_as_str(X_tr[c]), _as_str(X_va[c]), _as_str(X_te[c])], ignore_index=True)
            freq = all_vals.value_counts(normalize=True)
            for f in (X_tr, X_va, X_te):
                f[f"{c}__freq"] = _as_str(f[c]).map(freq).astype(float)
    return X_tr, X_va, X_te


# ----------------------------------------------------------------------------- public

def run_model(name: str, params: dict | None, X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame, X_test: pd.DataFrame, ctx: dict, features: dict | None = None):
    """Обучить модель из реестра и вернуть (val_pred, test_pred) по контракту модуля."""
    canon = normalize_model_name(name)
    if canon is None:
        raise ValueError(f"неизвестная модель '{name}'. Доступны: {list(available_models())}")
    if canon not in available_models():
        raise ImportError(f"модель '{canon}' недоступна в окружении. Доступны: {list(available_models())}")
    params = dict(params or {})
    if params.get("seed_offset"):  # бэггинг по сидам: та же конфигурация, другой случайный посев
        ctx = {**ctx, "seed": int(ctx.get("seed", 42)) + int(params["seed_offset"])}
    y_train = np.asarray(y_train)
    X_train, X_val, X_test = apply_feature_options(X_train, X_val, X_test, features)
    transform = params.get("target_transform")
    y_fit = y_train
    use_log = (not _is_clf(ctx)) and transform == "log1p" and float(np.min(y_train)) > -1.0
    if use_log:
        y_fit = np.log1p(y_train)
    val, test = _IMPL[canon](params, X_train, y_fit, X_val, X_test, ctx)
    if use_log:
        val, test = np.expm1(val), np.expm1(test)
    return val, test


def make_fit_predict(model: str, params: dict | None = None, features: dict | None = None):
    """fit_predict-функция для модели реестра (так харнесс одинаково запускает registry и LLM-код)."""
    def fit_predict(X_train, y_train, X_val, X_test, ctx):
        return run_model(model, params, X_train, y_train, X_val, X_test, ctx, features)
    return fit_predict
