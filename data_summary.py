import pandas as pd


def _get_target_profile(y: pd.Series, task_type: str) -> dict:
    result = {"missing": int(y.isna().sum())}

    is_classification_hint = "classif" in task_type.lower()
    is_categorical_target = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or (
                pd.api.types.is_numeric_dtype(y) and y.nunique() < 20)

    if is_classification_hint or is_categorical_target:
        counts = y.value_counts()
        result.update({
            "n_classes": int(y.nunique()),
            "class_counts": counts.head(20).to_dict(),
            "imbalance_ratio": (round(counts.max() / counts.min(), 2) if len(counts) > 1 else None),
        })
    else:
        numeric_y = pd.to_numeric(y, errors="coerce")
        result.update({
            "mean": numeric_y.mean(),
            "std": numeric_y.std(),
            "min": numeric_y.min(),
            "median": numeric_y.median(),
            "max": numeric_y.max(),
            "skewness": numeric_y.skew(),
        })
    return result


def _get_schema_profile(df: pd.DataFrame) -> dict:
    groups = {
        "numeric": df.select_dtypes(include="number").columns.tolist(),
        "boolean": df.select_dtypes(include="bool").columns.tolist(),
        "datetime": df.select_dtypes(include="datetime").columns.tolist(),
    }
    groups["categorical"] = [
        c for c in df.select_dtypes(include=["object", "category", "string"]).columns
    ]
    known = set(c for cols in groups.values() for c in cols)
    groups["other"] = [c for c in df.columns if c not in known]
    return {
        "counts": {k: len(v) for k, v in groups.items()},
        "examples": {k: v[:5] for k, v in groups.items()},
    }


def get_data_profile_summary(
        train: pd.DataFrame,
        test: pd.DataFrame | None,
        target: str,
        task_type: str,
        data_dictionary: pd.DataFrame | None = None,
        id_columns: list[str] | None = None,
) -> dict:
    if train.empty or target not in train.columns:
        raise ValueError("train должен содержать target и не быть пустым.")

    if id_columns is None and data_dictionary is not None:
        required = {"feature", "role"}
        if required.issubset(data_dictionary.columns):
            id_columns = data_dictionary.loc[data_dictionary["role"].eq("id"), "feature"].tolist()

    id_columns = [column for column in (id_columns or []) if column in train.columns and column != target]
    X = train.drop(columns=[target, *id_columns])

    missing = X.isna().sum()
    constants = X.columns[X.nunique(dropna=False) <= 1].tolist()
    warnings = []

    if constants:
        warnings.append({"code": "CONSTANT_FEATURES", "features": constants[:10]})
    if train[target].isna().any():
        warnings.append({"code": "MISSING_TARGET", "count": int(train[target].isna().sum())})

    train_test = None
    if test is not None:
        test_features = test.drop(columns=[column for column in id_columns if column in test.columns])
        train_test = {
            "train_only_columns": sorted(set(X.columns) - set(test_features.columns)),
            "test_only_columns": sorted(set(test_features.columns) - set(X.columns)),
            "new_categories": {},
        }
        for col in X.select_dtypes(include=["object", "category", "string"]).columns.intersection(
                test_features.columns):
            unseen = set(test_features[col].dropna()) - set(X[col].dropna())
            if unseen:
                train_test["new_categories"][col] = len(unseen)

    dictionary_summary = None
    if data_dictionary is not None:
        required = {"feature", "role", "dtype", "unit", "description_ru"}
        dictionary_summary = {"missing_fields": sorted(required - set(data_dictionary.columns))}
        if not dictionary_summary["missing_fields"]:
            dictionary_summary["roles"] = data_dictionary["role"].fillna("unknown").value_counts().to_dict()

    return {
        "dataset": {
            "train_rows": len(train),
            "test_rows": len(test) if test is not None else None,
            "n_features": len(X.columns),
            "id_columns": id_columns,
        },
        "schema": _get_schema_profile(X),
        "quality": {
            "missing_cells": int(missing.sum()),
            "missing_rate": round(float(missing.sum() / X.size), 4) if X.size else 0.0,
            "columns_with_missing": int((missing > 0).sum()),
            "duplicate_rows": int(train.duplicated().sum()),
            "constant_features": constants[:20],
        },
        "target": _get_target_profile(train[target], task_type),
        "train_test": train_test,
        "data_dictionary": dictionary_summary,
        "warnings": warnings,
    }