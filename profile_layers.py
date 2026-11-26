"""Трёхуровневое профилирование данных для ML-агента с ограниченным контекстом."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd
from data_profiler import DataProfiler
from data_summary import get_data_profile_summary

def build_profile_layers(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    target: str,
    task_type: str,
    metric: str,
    optimization_direction: str,
    data_dictionary: pd.DataFrame | None = None,
    compact_token_budget: int = 2_000,
) -> dict[str, dict[str, Any]]:
    profiler = DataProfiler(train, target, task_type, test, data_dictionary)
    feature_columns = list(profiler.X.columns)

    columns = {column: profiler.inspect_column(column, detail_level="full") for column in feature_columns}
    numeric_columns = profiler.X.select_dtypes(include="number").columns.tolist()

    shifts = profiler.analyze_train_test_shift(feature_columns) if test is not None else {}
    relationships = profiler.analyze_feature_relationships(numeric_columns)
    target_associations = _target_associations(profiler, numeric_columns)

    full_profile = {
        "task": {
            "target": target,
            "task_type": task_type,
            "metric": metric,
            "optimization_direction": optimization_direction,
        },
        "summary": get_data_profile_summary(train, test, target, task_type, data_dictionary, profiler.id_columns),
        "target": profiler.inspect_target(),
        "columns": columns,
        "duplicates": profiler.inspect_duplicates(),
        "suspicious_values": {column: profiler.inspect_suspicious_values(column) for column in feature_columns},
        "relationships": relationships,
        "target_associations": target_associations,
        "train_test_shift": shifts,
        "data_dictionary": profiler.get_data_dictionary_summary(),
        "dictionary_rows": {column: profiler.get_feature_metadata(column) for column in feature_columns} if data_dictionary is not None else {},
    }
    return {
        "full_profile": full_profile,
        "compact_context": build_compact_context(full_profile, compact_token_budget),
    }

def build_compact_context(full_profile: dict[str, Any], token_budget: int = 2_000) -> dict[str, Any]:
    summary = full_profile["summary"]
    columns = full_profile["columns"]

    missing = sorted(
        ({"feature": name, "missing_rate": details["missing_rate"]} for name, details in columns.items() if details["missing_rate"] >= 0.05),
        key=lambda item: item["missing_rate"], reverse=True,
    )[:10]

    high_cardinality = [
        {"feature": name, "unique_values": details["n_unique"]}
        for name, details in columns.items()
        if details["dtype"] in {"object", "string", "category"} and details["n_unique"] >= 50
    ][:10]

    shift_signals = _shift_signals(full_profile["train_test_shift"])
    suspicious = _suspicious_signals(full_profile["suspicious_values"])
    dictionary_features = _dictionary_features(full_profile, missing, shift_signals)

    context = {
        "task": full_profile["task"],
        "dataset": {**summary["dataset"], "schema": summary["schema"]["counts"]},
        "target": _compact_target_profile(full_profile["target"]),
        "data_quality": {
            "missing_cells": summary["quality"]["missing_cells"],
            "missing_rate": summary["quality"]["missing_rate"],
            "columns_with_missing": summary["quality"]["columns_with_missing"],
            "duplicate_feature_rows": full_profile["duplicates"]["duplicate_rows"],
            "constant_features": summary["quality"]["constant_features"],
            "high_missing_features": missing,
            "high_cardinality_categoricals": high_cardinality,
            "suspicious_values": suspicious,
        },
        "signals": {
            "strong_feature_correlations": full_profile["relationships"]["strong_correlations"][:10],
            "top_target_associations": full_profile["target_associations"][:10],
            "train_test_shift": shift_signals,
        },
        "data_dictionary": dictionary_features,
        "detail_access": {
            "available": ["column profile", "target association", "correlation matrix", "train/test shift", "duplicates", "suspicious values", "data dictionary search"],
            "instruction": "Request details only for a concrete modelling decision.",
        },
    }
    return _fit_context_to_token_budget(context, token_budget)

def save_profile_layers(layers: dict[str, dict[str, Any]], directory: str | Path) -> None:
    output_directory = Path(directory)
    # Создаем директорию, если её нет (это безопасно и не создает глубоких вложенных структур, если передан плоский путь)
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, payload in layers.items():
        (output_directory / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
        )

def _target_associations(profiler: DataProfiler, numeric_columns: list[str]) -> list[dict[str, Any]]:
    # УНИВЕРСАЛЬНАЯ ПРОВЕРКА: вместо task_type != "regression" проверяем, является ли target числовым
    y = profiler.train[profiler.target]
    is_numeric_target = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 20

    if not is_numeric_target:
        return []

    associations = []
    for column in numeric_columns:
        correlation = profiler.analyze_feature_target(column).get("pearson_correlation")
        if correlation is not None:
            associations.append({"feature": column, "pearson_correlation": round(float(correlation), 4)})

    return sorted(associations, key=lambda item: abs(item["pearson_correlation"]), reverse=True)

def _compact_target_profile(target_profile: dict[str, Any]) -> dict[str, Any]:
    result = dict(target_profile)
    class_counts = target_profile.get("class_counts")
    class_proportions = target_profile.get("class_proportions")
    if not isinstance(class_counts, dict):
        return result

    top_classes = sorted(class_counts, key=class_counts.__getitem__, reverse=True)[:15]
    result["class_counts"] = {class_name: class_counts[class_name] for class_name in top_classes}
    if isinstance(class_proportions, dict):
        result["class_proportions"] = {class_name: class_proportions[class_name] for class_name in top_classes if class_name in class_proportions}
    result["omitted_class_count"] = max(0, len(class_counts) - len(top_classes))
    return result

def _fit_context_to_token_budget(context: dict[str, Any], token_budget: int) -> dict[str, Any]:
    if token_budget <= 0:
        raise ValueError("token_budget должен быть положительным.")
    fitted = json.loads(json.dumps(context, ensure_ascii=False, default=str))
    shrink_paths = [
        ("data_dictionary",), ("signals", "train_test_shift"), ("signals", "top_target_associations"),
        ("signals", "strong_feature_correlations"), ("data_quality", "suspicious_values"),
        ("data_quality", "high_cardinality_categoricals"), ("data_quality", "high_missing_features"),
    ]
    while _estimate_tokens(fitted) > token_budget:
        changed = False
        for path in shrink_paths:
            collection = fitted
            for key in path:
                collection = collection[key]
            if collection:
                collection.pop()
                changed = True
                if _estimate_tokens(fitted) <= token_budget:
                    return fitted
        if not changed:
            break
    if _estimate_tokens(fitted) > token_budget:
        raise ValueError("Обязательный компактный контекст превышает token_budget. Увеличьте лимит контекста.")
    return fitted

def _estimate_tokens(payload: dict[str, Any]) -> int:
    serialized = json.dumps(payload, ensure_ascii=False, default=str)
    return max(1, (len(serialized) + 3) // 4)

def _shift_signals(shifts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    signals = []
    for feature, details in shifts.items():
        mean_shift = details.get("standardized_mean_difference", 0.0)
        missing_shift = details.get("missing_rate_difference", 0.0)
        unseen_share = details.get("unseen_category_row_share", 0.0)
        if mean_shift >= 0.5 or missing_shift >= 0.05 or unseen_share >= 0.01:
            signals.append({"feature": feature, **details})
    return sorted(signals, key=lambda item: max(item.get("standardized_mean_difference", 0.0), item.get("missing_rate_difference", 0.0), item.get("unseen_category_row_share", 0.0)), reverse=True)[:10]

def _suspicious_signals(checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    signals = []
    for feature, details in checks.items():
        sentinels = details.get("sentinel_candidates", {})
        markers = details.get("possible_missing_markers", {})
        if sentinels or markers:
            signals.append({"feature": feature, "sentinel_candidates": sentinels, "possible_missing_markers": markers})
    return signals[:10]

def _dictionary_features(full_profile: dict[str, Any], missing: list[dict[str, Any]], shift_signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dictionary = full_profile["data_dictionary"]
    if not dictionary["available"]:
        return []
    names = [item["feature"] for item in missing + shift_signals]
    for association in full_profile["target_associations"][:5]:
        names.append(association["feature"])
    dictionary_rows = full_profile.get("dictionary_rows", {})
    return [dictionary_rows[name] for name in dict.fromkeys(names) if name in dictionary_rows]