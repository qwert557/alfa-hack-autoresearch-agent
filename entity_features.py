"""entity_features.py — история объекта по времени (панельные данные).

Если в данных есть период (год, месяц...), а одни и те же объекты повторяются из периода в период
(здания, клиенты, магазины), то таргет объекта в прошлых периодах — сильнейший признак. Явного id
объекта обычно нет, поэтому агент восстанавливает его сам:

  * жадно подбирает набор неизменных признаков K (ключ объекта) так, чтобы
      U — внутри одного периода ключ почти уникален (один объект = одна строка),
      R — строки поздних периодов часто встречают свой ключ в более ранних периодах;
    максимизируется R·U; признаки, меняющиеся каждый период (погода и т.п.), ломают R и не берутся;
  * для каждой строки считает по СТРОГО более ранним периодам: последний таргет объекта, среднее,
    число наблюдений, разрыв в периодах. Свой таргет и таргеты того же/будущих периодов не используются,
    поэтому признак честный и в валидации «вперёд по времени», и на test.

Ничего не настроено под конкретный датасет: если объекты не повторяются (R мало) — признаки не строятся.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agent_utils import log

MIN_R, MIN_U = 0.2, 0.9


def _key_hash(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(df[cols], index=False).values


def _score(allx: pd.DataFrame, per: np.ndarray, cols: list[str], later: np.ndarray) -> tuple[float, float]:
    h = _key_hash(allx, cols)
    df = pd.DataFrame({"h": h, "p": per})
    u = df.drop_duplicates().shape[0] / len(df)
    first = df.groupby("h")["p"].transform("min").values
    r = float((per > first)[later].mean()) if later.any() else 0.0
    return r, u


def find_entity_key(allx: pd.DataFrame, per: np.ndarray, candidates: list[str], max_cols: int = 6,
                    time_limit: float = 60.0) -> tuple[list[str], float, float]:
    import time
    t0 = time.time()
    later = per > np.nanmin(per)
    key: list[str] = []
    best, best_r, best_u = 0.0, 0.0, 0.0
    for _ in range(max_cols):
        res = []
        for c in candidates:
            if c in key:
                continue
            r, u = _score(allx, per, key + [c], later)
            res.append((r * u, r, u, c))
            if time.time() - t0 > time_limit:
                break
        if not res:
            break
        s, r, u, c = max(res)
        if s <= best + 5e-4:
            break
        key.append(c)
        best, best_r, best_u = s, r, u
        if time.time() - t0 > time_limit:
            break
    return key, best_r, best_u


def _history(ent: np.ndarray, per: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[последнее значение, среднее, число наблюдений, разрыв] по строго более ранним периодам."""
    df = pd.DataFrame({"e": ent, "p": per, "y": y})
    g = df.dropna(subset=["y"]).groupby(["e", "p"], sort=True)["y"].agg(["sum", "count"]).reset_index()
    g["mean"] = g["sum"] / g["count"]
    gg = g.groupby("e", sort=False)
    g["cum_sum"] = gg["sum"].cumsum() - g["sum"]
    g["cum_cnt"] = gg["count"].cumsum() - g["count"]
    g["prev_mean"] = gg["mean"].shift(1)
    g["prev_p"] = gg["p"].shift(1)
    # для строки (e, p) нужна статистика по периодам < p: берём последнюю запись с p' < p
    rows = df[["e", "p"]].copy()
    rows["_i"] = np.arange(len(rows))
    stats = g[["e", "p", "sum", "count", "mean", "cum_sum", "cum_cnt"]].copy()
    stats["tot_sum"] = stats["cum_sum"] + stats["sum"]
    stats["tot_cnt"] = stats["cum_cnt"] + stats["count"]
    stats = stats.rename(columns={"p": "sp", "mean": "last_mean"})[["e", "sp", "last_mean", "tot_sum", "tot_cnt"]]
    rows = rows.sort_values("p")
    stats = stats.sort_values("sp")
    rows["p_key"] = rows["p"].astype(float) - 1e-9
    stats["sp"] = stats["sp"].astype(float)
    m = pd.merge_asof(rows, stats, left_on="p_key", right_on="sp", by="e", direction="backward",
                      allow_exact_matches=True)
    m = m.sort_values("_i")
    out = np.full((len(df), 4), np.nan)
    cnt = m["tot_cnt"].values.astype(float)
    has = ~np.isnan(cnt) & (cnt > 0)
    out[has, 0] = m["last_mean"].values[has]
    out[has, 1] = (m["tot_sum"].values[has] / cnt[has])
    out[:, 2] = np.where(has, cnt, 0.0)
    out[has, 3] = m["p"].values[has] - m["sp"].values[has]
    return out


def add_entity_features(X: pd.DataFrame, X_test: pd.DataFrame, y: np.ndarray, period_col: str,
                        task_type: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    info: dict = {"used": False}
    if period_col not in X.columns or period_col not in X_test.columns:
        return X, X_test, info
    per_tr = pd.to_numeric(X[period_col], errors="coerce").values.astype(float)
    per_te = pd.to_numeric(X_test[period_col], errors="coerce").values.astype(float)
    per = np.concatenate([per_tr, per_te])
    if np.isnan(per).mean() > 0.05 or len(np.unique(per[~np.isnan(per)])) > 200 or len(np.unique(per_tr)) < 2:
        return X, X_test, info
    allx = pd.concat([X, X_test], ignore_index=True)
    cands = [c for c in allx.columns if c != period_col and not c.startswith(f"{period_col}__")
             and allx[c].nunique(dropna=False) > 1]
    key, r, u = find_entity_key(allx, per, cands)
    info.update(key=key, R=round(r, 4), U=round(u, 4))
    if len(key) < 1 or r < MIN_R or u < MIN_U:
        log(f"История объектов: не найдена (ключ {key}, R={r:.2f}, U={u:.2f})")
        return X, X_test, info

    n = len(X)
    if task_type == "regression" or task_type == "binary_classification":
        yy = np.concatenate([np.asarray(y, float), np.full(len(X_test), np.nan)])
    else:
        yy = None
    keys = [key] + ([key[:2]] if len(key) >= 3 else [])
    new_cols = []
    X, X_test = X.copy(), X_test.copy()
    for k, kk in enumerate(keys):
        ent = _key_hash(allx, kk)
        tag = f"ent{k}"
        if yy is not None:
            h = _history(ent, per, yy)
            cols = {f"{tag}_last": h[:, 0], f"{tag}_mean": h[:, 1], f"{tag}_cnt": h[:, 2], f"{tag}_gap": h[:, 3]}
        else:
            # мультикласс: последняя метка объекта + доли классов в прошлом
            k_cls = int(np.max(y)) + 1
            cols = {}
            ylab = np.concatenate([np.asarray(y, float), np.full(len(X_test), np.nan)])
            base = None
            for c in range(min(k_cls, 20)):
                ind = np.where(np.isnan(ylab), np.nan, (ylab == c).astype(float))
                h = _history(ent, per, ind)
                cols[f"{tag}_share_c{c}"] = h[:, 1]
                if base is None:
                    base = h
            cols[f"{tag}_cnt"] = base[:, 2]
            cols[f"{tag}_gap"] = base[:, 3]
        for name, vals in cols.items():
            X[name] = vals[:n]
            X_test[name] = vals[n:]
            new_cols.append(name)
    cov_tr = float((X[f"ent0_cnt"] > 0).mean())
    cov_te = float((X_test[f"ent0_cnt"] > 0).mean())
    info.update(used=True, columns=new_cols, coverage_train=round(cov_tr, 3), coverage_test=round(cov_te, 3))
    log(f"История объектов по '{period_col}': ключ {key} (R={r:.2f}, U={u:.2f}); "
        f"признаков {len(new_cols)}, покрытие train {cov_tr:.0%}, test {cov_te:.0%}")
    return X, X_test, info
