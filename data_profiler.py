import numpy as np
import pandas as pd


class DataProfiler:
    def __init__(
            self,
            train: pd.DataFrame,
            target: str,
            task_type: str,
            test: pd.DataFrame | None = None,
            data_dictionary: pd.DataFrame | None = None,
            id_columns: list[str] | None = None,
    ):
        if target not in train.columns:
            raise ValueError(f"Target {target!r} отсутствует в train.")

        # УБРАНА ЖЕСТКАЯ ПРОВЕРКА task_type
        if not isinstance(task_type, str) or not task_type.strip():
            raise ValueError("task_type должен быть непустой строкой.")

        self.train = train
        self.test = test
        self.target = target
        self.task_type = task_type
        self.dictionary = data_dictionary

        if id_columns is None and data_dictionary is not None:
            required = {"feature", "role"}
            if required.issubset(data_dictionary.columns):
                id_columns = data_dictionary.loc[
                    data_dictionary["role"].eq("id"), "feature"
                ].tolist()

        self.id_columns = [
            column for column in (id_columns or [])
            if column in train.columns and column != target
        ]
        self.X = train.drop(columns=[target, *self.id_columns])

    @staticmethod
    def _json_value(value):
        if value is None or pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def inspect_column(self, column: str, detail_level: str = "brief") -> dict:
        if column not in self.X.columns:
            raise ValueError(f"Неизвестный признак: {column}")
        if detail_level not in ("brief", "full"):
            raise ValueError("detail_level должен быть brief или full.")

        s = self.X[column]
        counts = s.value_counts(dropna=False)
        result = {
            "feature": column,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "missing_count": int(s.isna().sum()),
            "missing_rate": round(float(s.isna().mean()), 4),
            "top_values": {str(k): int(v) for k, v in counts.head(10).items()},
        }

        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            result["statistics"] = {
                "mean": self._json_value(s.mean()),
                "std": self._json_value(s.std()),
                "min": self._json_value(s.min()),
                "median": self._json_value(s.median()),
                "max": self._json_value(s.max()),
            }
            if detail_level == "full":
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                result["quantiles"] = {
                    str(q): self._json_value(s.quantile(q))
                    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
                }
                result["iqr_outlier_count"] = int(
                    ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                )
        elif detail_level == "full":
            result["category_count"] = int(s.nunique(dropna=True))
            result["rare_values"] = {
                str(k): int(v)
                for k, v in counts[counts / max(len(s), 1) < 0.01].head(20).items()
            }
        return result

    def inspect_target(self) -> dict:
        y = self.train[self.target]
        result = {
            "target": self.target,
            "task_type": self.task_type,
            "missing_count": int(y.isna().sum()),
        }

        # ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ: если в названии задачи есть "classif" ИЛИ target категориальный/с малым числом уникальных значений
        is_classification_hint = "classif" in self.task_type.lower()
        is_categorical_target = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y) or (
                    pd.api.types.is_numeric_dtype(y) and y.nunique() < 20)

        if is_classification_hint or is_categorical_target:
            counts = y.value_counts()
            result.update({
                "n_classes": int(y.nunique()),
                "class_counts": {str(k): int(v) for k, v in counts.head(50).items()},
                "class_proportions": {str(k): round(float(v / len(y)), 4) for k, v in counts.head(50).items()},
                "imbalance_ratio": round(float(counts.max() / counts.min()), 2) if len(counts) > 1 else None,
            })
        else:
            # Для регрессии, ranking, time_series и т.д.
            numeric = pd.to_numeric(y, errors="coerce")
            result["non_numeric_count"] = int((y.notna() & numeric.isna()).sum())
            result["statistics"] = {
                "mean": self._json_value(numeric.mean()),
                "std": self._json_value(numeric.std()),
                "min": self._json_value(numeric.min()),
                "median": self._json_value(numeric.median()),
                "max": self._json_value(numeric.max()),
                "skewness": self._json_value(numeric.skew()),
            }
        return result

    def analyze_feature_target(self, feature: str) -> dict:
        if feature not in self.X.columns:
            raise ValueError(f"Неизвестный признак: {feature}")

        df = self.train[[feature, self.target]].dropna()
        s, y = df[feature], df[self.target]
        result = {"feature": feature, "target": self.target}
        numeric_feature = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)

        # Динамическая проверка вместо жесткой привязки к "regression"
        is_numeric_target = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) and y.nunique() > 20

        if is_numeric_target:
            y_num = pd.to_numeric(y, errors="coerce")
            if numeric_feature:
                result["pearson_correlation"] = self._json_value(s.corr(y_num))
            else:
                grouped = df.assign(_y=y_num).groupby(feature, observed=True)["_y"]
                result["target_by_category"] = {
                    str(k): {"count": int(v.count()), "mean": self._json_value(v.mean()),
                             "median": self._json_value(v.median())}
                    for k, v in list(grouped)[:30]
                }
        else:
            table = pd.crosstab(s, y, normalize="index")
            result["class_proportions_by_value"] = {
                str(k): {str(cls): round(float(val), 4) for cls, val in row.items()}
                for k, row in table.head(30).iterrows()
            }
        return result

    def analyze_feature_relationships(self, columns: list[str]) -> dict:
        missing = set(columns) - set(self.X.columns)
        if missing:
            raise ValueError(f"Неизвестные признаки: {sorted(missing)}")
        numeric = self.X[columns].select_dtypes(include="number")
        corr = numeric.corr(method="pearson")
        pairs = []
        for i, col_a in enumerate(corr.columns):
            for col_b in corr.columns[i + 1:]:
                value = corr.loc[col_a, col_b]
                if pd.notna(value) and abs(value) >= 0.8:
                    pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": round(float(value), 4)})
        return {"numeric_correlation_matrix": corr.round(4).to_dict(), "strong_correlations": pairs}

    def analyze_train_test_shift(self, columns: list[str]) -> dict:
        if self.test is None:
            raise ValueError("Для анализа необходим test.")
        result = {}
        for col in columns:
            if col not in self.X.columns or col not in self.test.columns:
                continue
            a, b = self.X[col], self.test[col]
            item = {"missing_rate_difference": round(float(abs(a.isna().mean() - b.isna().mean())), 4)}
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                scale = max(float(a.std()), float(b.std()), 1e-12)
                item["standardized_mean_difference"] = round(float(abs(a.mean() - b.mean()) / scale), 4)
            else:
                train_values = set(a.dropna().astype(str))
                test_values = set(b.dropna().astype(str))
                unseen = test_values - train_values
                item["unseen_categories"] = sorted(unseen)[:20]
                item["unseen_category_count"] = len(unseen)
                item["unseen_category_row_share"] = round(float(b.astype(str).isin(unseen).mean()), 4)
            result[col] = item
        return result

    def search_data_dictionary(self, query: str, limit: int = 10) -> dict:
        if self.dictionary is None:
            return {"results": [], "message": "Data Dictionary не задан."}
        fields = ["feature", "role", "dtype", "unit", "description_ru"]
        available = [c for c in fields if c in self.dictionary.columns]
        mask = pd.Series(False, index=self.dictionary.index)
        for col in available:
            mask |= self.dictionary[col].astype(str).str.contains(query, case=False, na=False, regex=False)
        return {
            "results": self.dictionary.loc[mask, available].head(limit).where(
                pd.notna(self.dictionary.loc[mask, available].head(limit)), None
            ).to_dict(orient="records")
        }

    def get_data_dictionary_summary(self) -> dict:
        if self.dictionary is None:
            return {"available": False, "reason": "Data Dictionary не задан."}
        result = {"available": True, "n_records": len(self.dictionary), "fields": list(self.dictionary.columns)}
        if "role" in self.dictionary:
            result["role_counts"] = self.dictionary["role"].fillna("<unknown>").value_counts().to_dict()
        return result

    def get_feature_metadata(self, feature: str) -> dict:
        if self.dictionary is None or "feature" not in self.dictionary:
            raise ValueError("Data Dictionary недоступен или не содержит feature.")
        rows = self.dictionary[self.dictionary["feature"].astype(str) == str(feature)]
        if rows.empty:
            raise ValueError(f"Признак {feature!r} не найден в словаре.")
        result = rows.iloc[0].where(pd.notna(rows.iloc[0]), None).to_dict()
        if feature in self.train.columns:
            result["actual_dtype"] = str(self.train[feature].dtype)
            result["missing_count"] = int(self.train[feature].isna().sum())
        return result

    def inspect_duplicates(self, columns: list[str] | None = None) -> dict:
        cols = columns or list(self.X.columns)
        missing = set(cols) - set(self.X.columns)
        if missing:
            raise ValueError(f"Неизвестные признаки: {sorted(missing)}")
        df = self.X[cols]
        duplicated = df.duplicated(keep=False)
        examples = df[duplicated].head(10)
        return {
            "columns_checked": cols,
            "duplicate_rows": int(df.duplicated().sum()),
            "rows_in_duplicate_groups": int(duplicated.sum()),
            "unique_combinations": int(len(df.drop_duplicates())),
            "examples": examples.astype(object).where(pd.notna(examples), None).to_dict(orient="records"),
        }

    def inspect_suspicious_values(self, column: str) -> dict:
        if column not in self.X.columns:
            raise ValueError(f"Неизвестный признак: {column}")
        s = self.X[column]
        result = {"feature": column, "dtype": str(s.dtype)}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            outliers = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
            sentinels = [-1, 999, -999, 9999, -9999]
            result.update({
                "iqr_outlier_count": int(outliers.count()),
                "iqr_outlier_examples": outliers.head(10).tolist(),
                "sentinel_candidates": {str(v): int((s == v).sum()) for v in sentinels if (s == v).any()},
            })
        else:
            counts = s.astype("string").value_counts()
            markers = {"unknown", "undefined", "n/a", "na", "null", "none", "?"}
            result["possible_missing_markers"] = {str(k): int(v) for k, v in counts.items() if
                                                  str(k).strip().lower() in markers}
            result["rare_values"] = {str(k): int(v) for k, v in counts.items() if v / max(len(s), 1) < 0.01}
        return result