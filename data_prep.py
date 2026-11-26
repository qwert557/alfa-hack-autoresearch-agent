"""data_prep.py — общая предобработка перед экспериментами (не зависит от гипотезы).

Делается один раз за запуск, результат кладётся в agent_work/cache/data.pkl и читается каждым
экспериментом-подпроцессом:
  * признаки = колонки, которые есть и в train, и в test (колонки только из train — потенциальные утечки);
  * константы и строковые id-подобные колонки выбрасываются;
  * даты разворачиваются в год/месяц/день/день недели/час/timestamp;
  * тексты заменяются длиной и числом слов (сложную обработку текста делает LLM в гипотезах kind=code);
  * bool → int; категориальные остаются строками — их кодирует каждая модель по-своему (model_zoo);
  * строки с пропущенным таргетом удаляются;
  * классификация: классы сортируются, таргет кодируется в 0..K-1 (исходные метки хранятся в meta).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from agent_utils import log
from entity_features import add_entity_features


def _expand_datetime(X: pd.DataFrame, X_test: pd.DataFrame, col: str) -> list[str]:
    new = []
    for df in (X, X_test):
        d = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}__year"] = d.dt.year
        df[f"{col}__month"] = d.dt.month
        df[f"{col}__day"] = d.dt.day
        df[f"{col}__dow"] = d.dt.dayofweek
        if (d.dt.hour.fillna(0) != 0).any():
            df[f"{col}__hour"] = d.dt.hour
        df[f"{col}__ts"] = (d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
    for c in X.columns:
        if c.startswith(f"{col}__"):
            new.append(c)
    return new


def prepare_data(cfg, train: pd.DataFrame, test: pd.DataFrame, profile: dict) -> dict:
    kinds = profile["kinds"]
    target = cfg.target
    train = train[train[target].notna()].reset_index(drop=True)
    feats = [c for c in profile["feature_columns"] if c not in kinds["constant"] and c not in kinds["id_like"]]
    X = train[feats].copy()
    X_test = test[feats].copy()
    dropped = kinds["constant"] + kinds["id_like"]
    for c in kinds["datetime"]:
        if c in X.columns:
            _expand_datetime(X, X_test, c)
            X = X.drop(columns=[c])
            X_test = X_test.drop(columns=[c])
    for c in kinds["text"]:
        if c in X.columns:
            for df in (X, X_test):
                s = df[c].astype(object).where(df[c].notna(), "").astype(str)
                df[f"{c}__len"] = s.str.len()
                df[f"{c}__words"] = s.str.split().str.len()
            X = X.drop(columns=[c])
            X_test = X_test.drop(columns=[c])
    for df in (X, X_test):
        for c in df.columns:
            if pd.api.types.is_bool_dtype(df[c]):
                df[c] = df[c].astype("int8")
    cat_features = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cat_features:  # единый тип: строки, пропуск = "__nan__"
        for df in (X, X_test):
            df[c] = df[c].astype(object).where(df[c].notna(), "__nan__").map(str)
    num_features = [c for c in X.columns if c not in cat_features]
    for c in num_features:
        X_test[c] = pd.to_numeric(X_test[c], errors="coerce")

    meta = {
        "task_type": cfg.task_type, "metric": cfg.metric, "prediction_format": cfg.prediction_format,
        "cat_features": cat_features, "num_features": num_features, "dropped_features": dropped,
        "n_train": len(X), "n_test": len(X_test),
    }
    if cfg.is_classification:
        y_raw = train[target]
        uniq = pd.unique(y_raw)
        try:
            classes = sorted(uniq.tolist())
        except TypeError:
            classes = sorted(uniq.tolist(), key=str)
        index = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(index).astype(int).values
        meta.update({"classes": classes, "n_classes": len(classes),
                     "class_counts": np.bincount(y, minlength=len(classes)).tolist()})
        if cfg.task_type == "binary_classification":
            pl = cfg.positive_label
            meta["positive_index"] = next((i for i, c in enumerate(classes) if str(c) == str(pl)), len(classes) - 1)
    else:
        y = pd.to_numeric(train[target], errors="coerce").astype(float).values
        meta.update({"n_classes": 0, "classes": None, "target_min": float(np.min(y)), "target_max": float(np.max(y))})
    temporal = profile.get("temporal") or {}
    if temporal and not temporal.get("is_datetime") and os.environ.get("AGENT_DISABLE_ENTITY", "0") != "1":
        try:
            X, X_test, ent_info = add_entity_features(X, X_test, y, temporal["column"], cfg.task_type)
            meta["entity_features"] = ent_info
            if ent_info.get("used"):
                num_features += [c for c in ent_info["columns"] if c not in num_features]
                meta["num_features"] = num_features
        except Exception as e:  # noqa: BLE001
            log(f"История объектов: ошибка {type(e).__name__}: {e} — продолжаю без неё")
    log(f"Подготовка данных: признаков {X.shape[1]} (кат. {len(cat_features)}, числ. {len(num_features)}), "
        f"выброшено {len(dropped)}, строк train {len(X)}")
    return {"X": X, "y": y, "X_test": X_test, "meta": meta,
            "test_ids": test[cfg.id_col].values if cfg.id_col and cfg.id_col in test.columns else None}
