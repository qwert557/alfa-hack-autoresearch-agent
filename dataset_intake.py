"""Обнаружение табличных датасетов и определение их ролей для ML-агента."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd

@dataclass(frozen=True)
class Inference:
    value: str | None
    confidence: str
    rationale: str

def discover_csv_datasets(root: str | Path) -> list[dict[str, str]]:
    root_path = Path(root)
    datasets = []
    for train_path in sorted(root_path.rglob("train.csv")):
        test_path = train_path.with_name("test.csv")
        if test_path.is_file():
            datasets.append({
                "directory": str(train_path.parent),
                "train_path": str(train_path),
                "test_path": str(test_path),
            })
    return datasets

def read_allowed_libraries(requirements_path: str | Path) -> list[str]:
    path = Path(requirements_path)
    if not path.is_file():
        raise FileNotFoundError(f"Не найден requirements.txt: {path}")
    libraries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        requirement = line.split("#", maxsplit=1)[0].strip()
        if requirement:
            libraries.append(requirement)
    return libraries

def infer_id_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_dictionary: pd.DataFrame | None = None,
) -> list[Inference]:
    dictionary_ids: set[str] = set()
    if data_dictionary is not None and {"feature", "role"}.issubset(data_dictionary.columns):
        dictionary_ids = set(data_dictionary.loc[
            data_dictionary["role"].eq("id"), "feature"
        ].astype(str))

    candidates = []
    for column in train.columns.intersection(test.columns):
        name = column.lower()
        uniqueness = train[column].nunique(dropna=False) / max(len(train), 1)
        if column in dictionary_ids:
            candidates.append(Inference(column, "high", "id role in data dictionary"))
        elif name in {"id", "index", "uuid"} or name.endswith("_id"):
            candidates.append(Inference(column, "medium", "identifier-like column name"))
        elif uniqueness >= 0.995:
            candidates.append(Inference(column, "low", "nearly unique values; may be an identifier"))
    return candidates

def build_intake_contract(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    task_type: str,
    metric: str,
    optimization_direction: str,
    data_dictionary: pd.DataFrame | None = None,
    requirements_path: str | Path | None = None,
) -> dict:
    if target not in train.columns:
        raise ValueError(f"Target {target!r} отсутствует в train.")
    if target in test.columns:
        raise ValueError(f"Target {target!r} не должен присутствовать в test.")

    # УБРАНА ЖЕСТКАЯ ПРОВЕРКА: теперь task_type может быть любой строкой (multiclass, ranking и т.д.)
    if not isinstance(task_type, str) or not task_type.strip():
        raise ValueError("task_type должен быть непустой строкой.")
    if not metric.strip():
        raise ValueError("metric не должна быть пустой.")
    if optimization_direction not in {"maximize", "minimize"}:
        raise ValueError("optimization_direction: только maximize или minimize.")

    id_candidates = infer_id_columns(train, test, data_dictionary)
    contract = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_columns": list(train.columns),
        "test_columns": list(test.columns),
        "task": {
            "target": target,
            "task_type": task_type,
            "metric": metric,
            "optimization_direction": optimization_direction,
        },
        "id_candidates": [asdict(candidate) for candidate in id_candidates],
    }
    if requirements_path is not None:
        contract["allowed_libraries"] = read_allowed_libraries(requirements_path)
    return contract