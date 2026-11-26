"""validation_plan.py — схема валидации выбирается ОДИН раз после EDA и не меняется.

Почему не даём LLM выбирать CV в каждой гипотезе: скоры разных схем несравнимы (holdout vs 5-fold),
а агент быстро «улучшит» метрику, выбрав валидацию попроще. Поэтому схема фиксирована на весь запуск,
а все эксперименты обучаются на одних и тех же фолдах (сохранены в folds.npz).

Правила выбора:
  1. EDA нашла временной сплит (test позже train) → валидация «вперёд по времени»:
     несколько последних периодов по очереди становятся валидацией, обучение — только на более ранних.
  2. Много дублей признаков (≥5%) → групповой сплит по хешу строки, чтобы копии одной строки
     не оказывались одновременно в train и val (иначе CV завышен).
  3. Иначе StratifiedKFold (классификация) / KFold (регрессия).
  Число фолдов: 5 до 200k строк, 3 до 1M, дальше 2; для редких классов уменьшается.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from agent_utils import log, read_json, write_json


@dataclass
class ValidationPlan:
    scheme: str
    n_splits: int
    description: str
    folds: list = field(default_factory=list)  # [(train_idx, val_idx), ...]
    details: dict = field(default_factory=dict)

    def save(self, work_dir: Path) -> None:
        arrays = {}
        for k, (tr, va) in enumerate(self.folds):
            arrays[f"train_{k}"] = np.asarray(tr, dtype=np.int64)
            arrays[f"val_{k}"] = np.asarray(va, dtype=np.int64)
        np.savez_compressed(Path(work_dir) / "folds.npz", **arrays)
        write_json(Path(work_dir) / "validation.json",
                   {"scheme": self.scheme, "n_splits": self.n_splits, "description": self.description,
                    "details": self.details})

    @staticmethod
    def load(work_dir: Path) -> "ValidationPlan":
        info = read_json(Path(work_dir) / "validation.json", {})
        data = np.load(Path(work_dir) / "folds.npz")
        n = info.get("n_splits", 0)
        folds = [(data[f"train_{k}"], data[f"val_{k}"]) for k in range(n)]
        return ValidationPlan(info.get("scheme", "?"), n, info.get("description", ""), folds, info.get("details", {}))


def _n_splits_for(n: int) -> int:
    return 5 if n <= 200_000 else 3 if n <= 1_000_000 else 2


def build_validation(cfg, X: pd.DataFrame, y: np.ndarray, profile: dict, seed: int = 42) -> ValidationPlan:
    n = len(X)
    temporal = profile.get("temporal")
    if temporal:
        col = temporal["column"]
        if temporal.get("is_datetime"):
            col = f"{col}__ts"
        if col in X.columns:
            t = pd.to_numeric(X[col], errors="coerce").values
            ok = ~np.isnan(t)
            periods = np.unique(t[ok])
            folds = []
            if len(periods) <= 100:
                val_periods = periods[-2:] if len(periods) >= 3 else periods[-1:]
                for p in val_periods:
                    tr = np.where(ok & (t < p))[0]
                    va = np.where(ok & (t == p))[0]
                    if len(tr) > 0 and len(va) > 0:
                        folds.append((tr, va))
                desc_p = ", ".join(str(int(p)) if float(p).is_integer() else str(p) for p in val_periods)
                desc = (f"Валидация вперёд по времени по '{col}': по очереди валидируемся на последних периодах "
                        f"({desc_p}), обучаясь только на более ранних. Test лежит после всех периодов train.")
            else:
                order = np.argsort(np.where(ok, t, -np.inf), kind="stable")
                for frac in (0.6, 0.8):
                    cut, end = int(n * frac), int(n * (frac + 0.2))
                    folds.append((np.sort(order[:cut]), np.sort(order[cut:end])))
                desc = (f"Валидация вперёд по времени по '{col}': два окна (обучение на первых 60%/80% по времени, "
                        f"валидация на следующих 20%).")
            if folds:
                plan = ValidationPlan("time_forward", len(folds), desc, folds,
                                      {"column": col, "val_sizes": [len(v) for _, v in folds]})
                log(f"Валидация: {desc}")
                return plan

    n_splits = _n_splits_for(n)
    is_clf = cfg.is_classification
    stratify = is_clf
    if is_clf:
        min_count = int(np.bincount(y).min()) if len(y) else 0
        if min_count < n_splits:
            if min_count >= 2:
                n_splits = min_count
            else:
                stratify = False
    dup_share = float(profile.get("duplicate_rows_share") or 0)
    use_groups = dup_share >= 0.05
    if use_groups:
        groups = pd.util.hash_pandas_object(X, index=False).values
        if stratify:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            folds = list(splitter.split(X, y, groups))
            scheme = "stratified_group_kfold"
        else:
            splitter = GroupKFold(n_splits=n_splits)
            folds = list(splitter.split(X, y, groups))
            scheme = "group_kfold"
        desc = (f"{scheme} ({n_splits} фолдов) с группами по хешу строки признаков: {dup_share:.1%} строк train — "
                f"дубли, копии одной строки держим в одном фолде, чтобы не завышать CV.")
    else:
        if stratify:
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y))
            scheme = "stratified_kfold"
        else:
            folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X))
            scheme = "kfold"
        desc = f"{scheme}, {n_splits} фолдов, shuffle, seed={seed}."
    plan = ValidationPlan(scheme, n_splits, desc, [(np.asarray(a), np.asarray(b)) for a, b in folds],
                          {"duplicate_rows_share": dup_share})
    log(f"Валидация: {desc}")
    return plan
