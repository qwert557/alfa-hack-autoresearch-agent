"""eda.py — задача 2: первичный анализ данных.

profile_dataset() собирает факты, которые реально влияют на решения агента:
  * типы признаков (числовые / категориальные / даты / тексты / id-подобные / константы);
  * пропуски в train и test и их сдвиг;
  * категориальные: кардинальность, редкие и новые (только в test) категории;
  * таргет: дисбаланс классов или распределение (асимметрия, выбросы);
  * дубликаты признаков и дубли с разными метками (потолок качества, утечка между фолдами);
  * колонки, которых нет в test (возможные утечки — в модель не идут);
  * временной сплит: признак, по которому test целиком лежит «в будущем» относительно train;
  * дрейф train→test: одномерный AUC по каждому признаку и adversarial validation;
  * сильно коррелированные пары.
summarize_for_llm() превращает профиль в короткий текст для промпта гипотез.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from agent_utils import log, short


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def _sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed)


def _looks_datetime(s: pd.Series) -> bool:
    vals = s.dropna().astype(str)
    if len(vals) == 0:
        return False
    vals = vals.sample(min(500, len(vals)), random_state=0)
    if not vals.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}|\d{4}-\d{2}", regex=True).mean() > 0.9:
        return False
    parsed = pd.to_datetime(vals, errors="coerce")
    return parsed.notna().mean() > 0.9


def classify_columns(train: pd.DataFrame, features: list[str]) -> dict:
    kinds = {"numeric": [], "categorical": [], "datetime": [], "text": [], "id_like": [], "constant": []}
    n = len(train)
    for c in features:
        s = train[c]
        nun = s.nunique(dropna=True)
        if nun <= 1:
            kinds["constant"].append(c)
        elif _is_num(s):
            kinds["numeric"].append(c)
        else:
            vals = s.dropna().astype(str)
            avg_len = float(vals.str.len().mean()) if len(vals) else 0.0
            if _looks_datetime(s):
                kinds["datetime"].append(c)
            elif nun / max(n, 1) > 0.95 and avg_len >= 6:
                kinds["id_like"].append(c)
            elif avg_len > 30 and nun / max(n, 1) > 0.2:
                kinds["text"].append(c)
            else:
                kinds["categorical"].append(c)
    return kinds


def detect_temporal(train: pd.DataFrame, test: pd.DataFrame, kinds: dict, exclude: set) -> dict | None:
    """Колонка, по которой test целиком (≥95%) позже всех строк train."""
    found = []
    for c in kinds["numeric"] + kinds["datetime"]:
        if c in exclude:
            continue
        tr, te = train[c], test[c]
        if c in kinds["datetime"]:
            tr, te = pd.to_datetime(tr, errors="coerce"), pd.to_datetime(te, errors="coerce")
        tr, te = tr.dropna(), te.dropna()
        if len(tr) < 10 or len(te) < 10 or tr.nunique() < 2:
            continue
        share_after = float((te > tr.max()).mean())
        if share_after >= 0.95:
            name_bonus = bool(re.search(r"year|date|time|period|month|week|day|год|дата|period|factor", c.lower()))
            found.append((not name_bonus, tr.nunique(), c, share_after, c in kinds["datetime"]))
    if not found:
        return None
    found.sort()
    _, nun, c, share, is_dt = found[0]
    return {"column": c, "n_periods_train": int(nun), "share_test_after_train": round(share, 3),
            "is_datetime": is_dt, "kind": "period" if nun <= 100 else "continuous"}


def _encode_for_drift(s_tr: pd.Series, s_te: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    if _is_num(s_tr):
        a = pd.to_numeric(s_tr, errors="coerce").astype(float)
        b = pd.to_numeric(s_te, errors="coerce").astype(float)
        fill = np.nanmin(np.concatenate([a.values, b.values])) - 1 if (a.notna().any() or b.notna().any()) else 0
        return a.fillna(fill).values, b.fillna(fill).values
    both = pd.concat([s_tr, s_te]).astype(object).where(lambda x: x.notna(), "__nan__").astype(str)
    freq = both.value_counts(normalize=True)
    enc = both.map(freq).astype(float).values
    return enc[: len(s_tr)], enc[len(s_tr):]


def drift_report(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], n: int = 20000) -> dict:
    from sklearn.metrics import roc_auc_score

    tr, te = _sample(train[cols], n, 1), _sample(test[cols], n, 2)
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    uni = []
    for c in cols:
        try:
            a, b = _encode_for_drift(tr[c], te[c])
            auc = roc_auc_score(y, np.r_[a, b])
            uni.append((abs(auc - 0.5), c, round(float(auc), 3)))
        except Exception:  # noqa: BLE001
            continue
    uni.sort(reverse=True)
    out = {"univariate_top": [{"feature": c, "auc": a} for _, c, a in uni[:10]]}
    # adversarial validation (одна модель, holdout)
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        X = pd.concat([tr, te], ignore_index=True)
        mask = []
        for c in X.columns:
            if _is_num(X[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
                mask.append(False)
            else:
                codes = pd.Categorical(X[c].astype(object).where(X[c].notna(), "__nan__").astype(str)).codes
                X[c] = codes.astype(float)
                mask.append(X[c].nunique() < 250)
        Xa, Xb, ya, yb = train_test_split(X.values, y, test_size=0.3, random_state=0, stratify=y)
        clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, categorical_features=np.array(mask),
                                             random_state=0)
        clf.fit(Xa, ya)
        out["adversarial_auc"] = round(float(roc_auc_score(yb, clf.predict_proba(Xb)[:, 1])), 3)
    except Exception as e:  # noqa: BLE001
        out["adversarial_auc"] = None
        out["adversarial_error"] = short(e, 200)
    return out


def profile_dataset(cfg, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    t0 = time.time()
    target, id_col = cfg.target, cfg.id_col
    features = [c for c in test.columns if c != id_col and c in train.columns and c != target]
    train_only = [c for c in train.columns if c not in test.columns and c != target]
    kinds = classify_columns(train, features)
    prof: dict = {
        "n_train": len(train), "n_test": len(test), "n_features": len(features),
        "feature_columns": features, "train_only_columns": train_only, "kinds": kinds,
    }
    miss = []
    for c in features:
        a, b = float(train[c].isna().mean()), float(test[c].isna().mean())
        if a > 0 or b > 0:
            miss.append({"feature": c, "train": round(a, 3), "test": round(b, 3), "shift": round(b - a, 3)})
    miss.sort(key=lambda r: -max(r["train"], r["test"]))
    prof["missing"] = miss[:25]
    prof["n_features_with_missing"] = len(miss)
    cats = []
    for c in kinds["categorical"]:
        s_tr = train[c].astype(object).where(train[c].notna(), "__nan__").astype(str)
        s_te = test[c].astype(object).where(test[c].notna(), "__nan__").astype(str)
        vc = s_tr.value_counts(normalize=True)
        rare = vc[vc < 0.005]
        cats.append({
            "feature": c, "nunique": int(len(vc)), "top_share": round(float(vc.iloc[0]), 3),
            "rare_categories": int(len(rare)), "rare_rows_share": round(float(rare.sum()), 4),
            "unseen_in_test_share": round(float((~s_te.isin(vc.index)).mean()), 4),
        })
    prof["categorical"] = cats

    y = train[target]
    prof["target_missing"] = int(y.isna().sum())
    y = y.dropna()
    if cfg.task_type == "regression":
        yv = y.astype(float)
        q = yv.quantile([0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]).round(4).tolist()
        iqr = q[4] - q[2]
        prof["target"] = {
            "mean": round(float(yv.mean()), 4), "std": round(float(yv.std()), 4),
            "quantiles_0_1_25_50_75_99_100": q, "skew": round(float(yv.skew()), 3),
            "share_negative": round(float((yv < 0).mean()), 4), "share_zero": round(float((yv == 0).mean()), 4),
            "outlier_share_3iqr": round(float(((yv > q[4] + 3 * iqr) | (yv < q[2] - 3 * iqr)).mean()), 4),
        }
    else:
        vc = y.astype(object).map(str).value_counts()
        prof["target"] = {
            "n_classes": int(len(vc)), "counts": {k: int(v) for k, v in vc.head(30).items()},
            "imbalance_ratio": round(float(vc.max() / max(vc.min(), 1)), 2), "min_count": int(vc.min()),
            "minority_classes": [k for k, v in vc.items() if v / len(y) < 0.02][:15],
        }
    # дубли
    X = train[features]
    try:
        h_tr = pd.util.hash_pandas_object(X, index=False)
        dup = h_tr.duplicated(keep=False)
        prof["duplicate_rows_share"] = round(float(h_tr.duplicated().mean()), 4)
        if dup.any():
            g = pd.DataFrame({"h": h_tr.values, "y": train[target].astype(str).values})[dup.values]
            conf = g.groupby("h")["y"].nunique()
            prof["conflicting_duplicate_groups"] = int((conf > 1).sum())
            prof["rows_in_conflicting_groups_share"] = round(float(g["h"].isin(conf[conf > 1].index).sum() / len(train)), 4)
        else:
            prof["conflicting_duplicate_groups"] = 0
            prof["rows_in_conflicting_groups_share"] = 0.0
        h_te = pd.util.hash_pandas_object(test[features], index=False)
        prof["test_rows_seen_in_train_share"] = round(float(h_te.isin(set(h_tr.values)).mean()), 4)
    except Exception as e:  # noqa: BLE001
        prof["duplicates_error"] = short(e, 200)
    # временной сплит
    prof["temporal"] = detect_temporal(train, test, kinds, exclude={id_col} if id_col else set())
    # дрейф
    drift_cols = kinds["numeric"] + kinds["categorical"]
    if drift_cols:
        prof["drift"] = drift_report(train, test, drift_cols)
    # корреляции
    num = kinds["numeric"]
    if 2 <= len(num) <= 400:
        s = _sample(train[num], 20000).astype(float)
        corr = s.corr().abs()
        pairs = []
        cols = corr.columns
        vals = corr.values
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if vals[i, j] > 0.98:
                    pairs.append((round(float(vals[i, j]), 4), cols[i], cols[j]))
        pairs.sort(reverse=True)
        prof["high_corr_pairs"] = [{"a": a, "b": b, "r": r} for r, a, b in pairs[:12]]
        prof["n_high_corr_pairs"] = len(pairs)
    prof["eda_seconds"] = round(time.time() - t0, 1)
    log(f"EDA за {prof['eda_seconds']}с: {len(features)} признаков, temporal={prof['temporal'] and prof['temporal']['column']}, "
        f"дубли={prof.get('duplicate_rows_share')}, adversarial_auc={prof.get('drift', {}).get('adversarial_auc')}")
    return prof


def summarize_for_llm(prof: dict, cfg) -> str:
    k = prof["kinds"]
    lines = [
        f"- train {prof['n_train']} строк, test {prof['n_test']}, признаков {prof['n_features']}: "
        f"числовых {len(k['numeric'])}, категориальных {len(k['categorical'])}, дат {len(k['datetime'])}, "
        f"текстов {len(k['text'])}, id-подобных {len(k['id_like'])}, констант {len(k['constant'])}.",
    ]
    if prof["train_only_columns"]:
        lines.append(f"- колонки только в train (в модель не идут): {prof['train_only_columns'][:10]}")
    t = prof["target"]
    if cfg.task_type == "regression":
        lines.append(f"- таргет: mean {t['mean']}, std {t['std']}, квантили 0/1/25/50/75/99/100% = "
                     f"{t['quantiles_0_1_25_50_75_99_100']}, skew {t['skew']}, отрицательных {t['share_negative']:.1%}, "
                     f"выбросов (3 IQR) {t['outlier_share_3iqr']:.1%}")
    else:
        lines.append(f"- классов {t['n_classes']}, дисбаланс max/min = {t['imbalance_ratio']}, "
                     f"min класс {t['min_count']} строк; распределение: {dict(list(t['counts'].items())[:12])}")
    if prof.get("missing"):
        top = ", ".join(f"{m['feature']} {m['train']:.0%}→{m['test']:.0%}" for m in prof["missing"][:8])
        lines.append(f"- пропуски (train→test): {top}; всего признаков с пропусками {prof['n_features_with_missing']}")
    else:
        lines.append("- пропусков нет")
    if prof.get("categorical"):
        top = ", ".join(f"{c['feature']}({c['nunique']} кат., новых в test {c['unseen_in_test_share']:.1%})"
                        for c in sorted(prof["categorical"], key=lambda c: -c["nunique"])[:8])
        lines.append(f"- категориальные: {top}")
    if prof.get("duplicate_rows_share"):
        lines.append(f"- дубли признаков в train: {prof['duplicate_rows_share']:.1%}; групп дублей с разными таргетами: "
                     f"{prof.get('conflicting_duplicate_groups')} ({prof.get('rows_in_conflicting_groups_share', 0):.1%} строк); "
                     f"строк test, точно совпадающих с train: {prof.get('test_rows_seen_in_train_share', 0):.1%}")
    if prof.get("temporal"):
        tt = prof["temporal"]
        lines.append(f"- ВРЕМЕННОЙ СПЛИТ: test лежит после train по '{tt['column']}' "
                     f"({tt['share_test_after_train']:.0%} строк test позже train, периодов в train {tt['n_periods_train']})")
    d = prof.get("drift") or {}
    if d:
        top = ", ".join(f"{u['feature']}={u['auc']}" for u in d.get("univariate_top", [])[:6])
        lines.append(f"- дрейф train→test: adversarial AUC {d.get('adversarial_auc')}; одномерный AUC: {top}")
    if prof.get("n_high_corr_pairs"):
        top = ", ".join(f"{p['a']}~{p['b']}" for p in prof["high_corr_pairs"][:5])
        lines.append(f"- сильно коррелированных пар (|r|>0.98): {prof['n_high_corr_pairs']}: {top}")
    if cfg.notes:
        lines.append("- из условия: " + "; ".join(cfg.notes[:6]))
    return "\n".join(lines)
