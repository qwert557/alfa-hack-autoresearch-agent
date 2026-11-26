"""task_understanding.py — задача 1: исследование задачи.

Что делает:
  1. находит файлы задачи (train/test/sample_submission/data_dictionary) — по путям из task.md, а если их
     нет, рекурсивным поиском рядом с task.md (на публичных данных пути в task.md не совпадают с
     реальной раскладкой);
  2. строит эвристическую постановку по данным: таргет = колонка train, которой нет в test; id = первая
     колонка sample_submission; тип задачи, метрика и формат ответа — по тексту и данным;
  3. просит LLM уточнить постановку по тексту условия и принимает её ответ только там, где он
     согласуется с данными (таргет существует, метрика известна, формат совместим с sample_submission);
  4. сохраняет task_config.json единого формата.

Единая схема task_config.json (объединяет форматы из чата команды):
    target, id_col, task_type, metric, optimization_direction, prediction_format, positive_label,
    submission_columns, task_description, notes, пути к файлам
и для совместимости со старым кодом дублирует ключи: target_col, metric_name.
load_task_config() понимает любой из вариантов.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import prompts
from agent_utils import log, read_json, short, write_json
from task_metrics import default_metric, detect_metric_in_text, resolve_metric

TASK_TYPES = ("regression", "binary_classification", "multiclass_classification")
FORMATS = ("value", "label", "proba_positive", "proba_per_class")


@dataclass
class TaskConfig:
    task_md: str
    task_dir: str
    train_path: str
    test_path: str
    sample_path: str | None
    dictionary_path: str | None
    submission_path: str
    target: str
    id_col: str | None
    task_type: str
    metric: str
    optimization_direction: str
    prediction_format: str
    submission_columns: list
    positive_label: object = None
    task_description: str = ""
    notes: list = field(default_factory=list)
    source: dict = field(default_factory=dict)

    @property
    def is_classification(self) -> bool:
        return self.task_type != "regression"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_col"] = self.target       # совместимость со старым ml_experiment / ml_analysis
        d["metric_name"] = self.metric
        return d


# ----------------------------------------------------------------------------- files

def read_table(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
        return df.head(nrows) if nrows else df
    return pd.read_csv(path, low_memory=False, nrows=nrows)


def _candidates(root: Path, max_depth: int = 4) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".csv", ".parquet") and len(p.relative_to(root).parts) <= max_depth:
            if "agent_work" in p.parts:
                continue
            out.append(p)
    return out


def locate_files(task_md: Path, text: str) -> dict:
    task_dir = task_md.parent
    found: dict[str, Path | None] = {"train": None, "test": None, "sample": None, "dictionary": None}

    def role(name: str) -> str | None:
        n = name.lower()
        if "sample" in n or ("submission" in n and n not in ("submission.csv", "submission.parquet")):
            return "sample"
        if "dictionary" in n or "description" in n or "columns" in n:
            return "dictionary"
        if "train" in n:
            return "train"
        if "test" in n:
            return "test"
        return None

    # 1) пути, упомянутые в task.md
    for m in re.findall(r"[\w./\\-]+\.(?:csv|parquet)", text):
        p = (task_dir / m.replace("\\", "/")).resolve()
        r = role(p.name)
        if r and p.exists() and found[r] is None:
            found[r] = p
    # 2) поиск рядом с task.md (короткие пути и точные имена — в приоритете)
    cands = sorted(_candidates(task_dir), key=lambda p: (len(p.relative_to(task_dir).parts), len(p.name)))
    exact = {"train": ("train.csv", "train.parquet"), "test": ("test.csv", "test.parquet"),
             "sample": ("sample_submission.csv", "sample_submission.parquet"),
             "dictionary": ("data_dictionary.csv",)}
    for r, names in exact.items():
        if found[r] is None:
            for p in cands:
                if p.name.lower() in names:
                    found[r] = p
                    break
    for p in cands:
        r = role(p.name)
        if r and found[r] is None and p.name.lower() != "submission.csv":
            found[r] = p
    if found["train"] is None or found["test"] is None:
        raise FileNotFoundError(f"не нашёл train/test рядом с {task_md}: {[str(c) for c in cands]}")
    return {k: (str(v) if v else None) for k, v in found.items()}


# ----------------------------------------------------------------------------- heuristics

def _sorted_classes(values) -> list:
    uniq = pd.unique(pd.Series(values).dropna())
    try:
        return sorted(uniq.tolist())
    except TypeError:
        return sorted(uniq.tolist(), key=str)


def _positive_label(classes: list, y: pd.Series):
    norm = {str(c).strip().lower(): c for c in classes}
    for key in ("1", "1.0", "true", "yes", "positive", "pos", "да", "fraud", "default", "churn", "attack"):
        if key in norm:
            return norm[key]
    counts = y.value_counts()
    return counts.index[-1]  # миноритарный класс


def heuristic_config(text: str, files: dict, train: pd.DataFrame, test_cols: list, sample: pd.DataFrame | None) -> dict:
    low = text.lower()
    train_cols = list(train.columns)
    # id
    id_col = None
    if sample is not None and len(sample.columns) and sample.columns[0] in test_cols:
        id_col = sample.columns[0]
    if id_col is None:
        for c in test_cols:
            if c.lower() in ("id", "row_id", "index", "uid") or c.lower().endswith("_id"):
                id_col = c
                break
    # target
    cands = [c for c in train_cols if c not in test_cols and c != id_col]
    target = None
    if sample is not None:
        for c in sample.columns[1:]:
            if c in cands:
                target = c
                break
    if target is None and cands:
        target = cands[0] if len(cands) == 1 else max(cands, key=lambda c: (c.lower() in low, -train_cols.index(c)))
    if target is None:
        raise ValueError("не удалось определить таргет: в train нет колонок, отсутствующих в test")
    y = train[target]
    nunique = y.nunique(dropna=True)
    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)

    metric_name = detect_metric_in_text(text)
    m = resolve_metric(metric_name) if metric_name else None
    reg_words = bool(re.search(r"регресс|regression", low))
    clf_words = bool(re.search(r"классифик|classification", low))
    if m is not None and m.task == "regression":
        task_type = "regression"
    elif m is not None and m.task in ("classification", "binary"):
        task_type = "binary_classification" if nunique <= 2 else "multiclass_classification"
    elif not numeric or nunique <= 2:
        task_type = "binary_classification" if nunique <= 2 else "multiclass_classification"
    elif clf_words and not reg_words and nunique <= 50:
        task_type = "multiclass_classification"
    else:
        task_type = "regression"
    if task_type == "binary_classification" and nunique > 2:
        task_type = "multiclass_classification"

    sub_cols = list(sample.columns) if sample is not None else [c for c in (id_col, target) if c]
    fmt = "value"
    positive = None
    if task_type != "regression":
        classes = _sorted_classes(y)
        class_names = {str(c) for c in classes}
        extra_cols = [c for c in sub_cols if c != id_col]
        if len(extra_cols) > 1 and {str(c) for c in extra_cols} <= class_names:
            fmt = "proba_per_class"
        elif m is not None and m.needs == "proba" and task_type == "binary_classification":
            fmt = "proba_positive"
        elif task_type == "binary_classification" and re.search(r"вероятност|probabilit", low):
            fmt = "proba_positive"
        else:
            fmt = "label"
        if sample is not None and fmt == "label" and task_type == "binary_classification" and len(sample.columns) > 1:
            sv = pd.to_numeric(sample.iloc[:, 1], errors="coerce")
            if sv.notna().all() and ((sv > 0) & (sv < 1)).any():
                fmt = "proba_positive"
        if task_type == "binary_classification":
            positive = _positive_label(classes, y)
    if m is None:
        m = default_metric(task_type, fmt)
    return {
        "target": target, "id_col": id_col, "task_type": task_type, "metric": m.name,
        "optimization_direction": m.direction, "prediction_format": fmt,
        "positive_label": positive, "submission_columns": sub_cols,
        "task_description": "", "notes": [],
    }


# ----------------------------------------------------------------------------- LLM

def _columns_summary(train: pd.DataFrame, limit: int = 80) -> str:
    rows = []
    for c in train.columns[:limit]:
        s = train[c]
        rows.append(f"{c}: {s.dtype}, uniq={s.nunique(dropna=True)}, nan={s.isna().mean():.1%}")
    if len(train.columns) > limit:
        rows.append(f"... и ещё {len(train.columns) - limit} колонок")
    return "\n".join(rows)


def _merge_llm(h: dict, llm_cfg: dict, train: pd.DataFrame, test_cols: list, sample: pd.DataFrame | None) -> tuple[dict, dict]:
    cfg = dict(h)
    source = {k: "heuristic" for k in h}
    t = llm_cfg.get("target")
    if t in train.columns and t not in test_cols:
        cfg["target"] = t
        source["target"] = "llm"
    idc = llm_cfg.get("id_col")
    if idc is None or idc in test_cols:
        if idc is not None:
            cfg["id_col"] = idc
            source["id_col"] = "llm"
    y = train[cfg["target"]]
    tt = llm_cfg.get("task_type")
    if tt in TASK_TYPES:
        numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
        nun = y.nunique(dropna=True)
        ok = not (tt == "regression" and not numeric) and not (tt == "binary_classification" and nun > 2)
        if ok:
            cfg["task_type"] = tt
            source["task_type"] = "llm"
    mname = llm_cfg.get("metric")
    m = resolve_metric(mname, cfg["task_type"]) if mname else None
    if m is not None:
        compatible = not (m.task == "regression" and cfg["task_type"] != "regression") and \
            not (m.task in ("classification", "binary") and cfg["task_type"] == "regression")
        if compatible:
            cfg["metric"] = m.name
            cfg["optimization_direction"] = m.direction
            source["metric"] = "llm"
    fmt = llm_cfg.get("prediction_format")
    if fmt in FORMATS:
        if cfg["task_type"] == "regression":
            fmt = "value"
        elif fmt == "value":
            fmt = cfg["prediction_format"]
        if fmt == "proba_per_class":
            classes = {str(c) for c in _sorted_classes(y)}
            extra = [c for c in cfg["submission_columns"] if c != cfg["id_col"]]
            if not (len(extra) > 1 and {str(c) for c in extra} <= classes):
                fmt = cfg["prediction_format"]
        if fmt == "proba_positive" and cfg["task_type"] != "binary_classification":
            fmt = cfg["prediction_format"]
        if fmt != cfg["prediction_format"]:
            source["prediction_format"] = "llm"
        cfg["prediction_format"] = fmt
    if cfg["task_type"] == "binary_classification":
        classes = _sorted_classes(y)
        pl = llm_cfg.get("positive_label")
        match = [c for c in classes if str(c) == str(pl)]
        cfg["positive_label"] = match[0] if match else (cfg.get("positive_label") if cfg.get("positive_label") in classes
                                                         else _positive_label(classes, y))
    else:
        cfg["positive_label"] = None
    if cfg["task_type"] == "regression":
        cfg["prediction_format"] = "value"
    elif cfg["prediction_format"] == "value":
        cfg["prediction_format"] = "label"
    if isinstance(llm_cfg.get("task_description"), str):
        cfg["task_description"] = llm_cfg["task_description"][:600]
    if isinstance(llm_cfg.get("notes"), list):
        cfg["notes"] = [short(n, 300) for n in llm_cfg["notes"][:12]]
    return cfg, source


def load_task_data(task_md: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Найти и прочитать файлы задачи целиком: (files, train, test, sample)."""
    task_md = Path(task_md).resolve()
    text = task_md.read_text(encoding="utf-8", errors="replace")
    files = locate_files(task_md, text)
    log(f"Файлы задачи: {json.dumps({k: (Path(v).name if v else None) for k, v in files.items()}, ensure_ascii=False)}")
    train = read_table(files["train"])
    test = read_table(files["test"])
    sample = read_table(files["sample"]) if files.get("sample") else None
    log(f"train {train.shape}, test {test.shape}" + (f", sample {sample.shape}" if sample is not None else ""))
    return files, train, test, sample


def understand_task(task_md: Path, llm=None, work_dir: Path | None = None, data: tuple | None = None) -> TaskConfig:
    """Определить постановку задачи. data = (files, train, test, sample) из load_task_data (чтобы не читать дважды)."""
    task_md = Path(task_md).resolve()
    text = task_md.read_text(encoding="utf-8", errors="replace")
    files, train_head, test_df, sample = data if data is not None else load_task_data(task_md)
    test_cols = list(test_df.columns)

    h = heuristic_config(text, files, train_head, test_cols, sample)
    log(f"Эвристика: target={h['target']}, id={h['id_col']}, type={h['task_type']}, metric={h['metric']}, "
        f"format={h['prediction_format']}")
    cfg_dict, source = dict(h), {k: "heuristic" for k in h}

    if llm is not None and llm.available():
        dict_text = ""
        if files.get("dictionary"):
            try:
                dict_text = Path(files["dictionary"]).read_text(encoding="utf-8", errors="replace")[:4000]
            except OSError:
                pass
        user = prompts.TASK_USER.format(
            task_text=text[:14000] + (f"\n\n## data_dictionary (начало)\n{dict_text}" if dict_text else ""),
            files=json.dumps({k: (str(Path(v).relative_to(task_md.parent)) if v else None) for k, v in files.items()},
                             ensure_ascii=False),
            train_columns=_columns_summary(train_head),
            target_candidates=", ".join(c for c in train_head.columns if c not in test_cols) or "—",
            sample_head=sample.head(3).to_string(index=False) if sample is not None else "нет sample_submission",
            heuristic=json.dumps(h, ensure_ascii=False, default=str),
        )
        llm_cfg, _ = llm.ask_json(prompts.TASK_SYSTEM, user, max_tokens=1200, temperature=0.1, stage="task",
                                  required_keys=("target", "task_type", "metric"))
        if llm_cfg:
            cfg_dict, source = _merge_llm(h, llm_cfg, train_head, test_cols, sample)
            diffs = {k: (h[k], cfg_dict[k]) for k in ("target", "task_type", "metric", "prediction_format")
                     if h[k] != cfg_dict[k]}
            log(f"LLM уточнила постановку: {diffs or 'совпадает с эвристикой'}")

    cfg = TaskConfig(
        task_md=str(task_md), task_dir=str(task_md.parent),
        train_path=files["train"], test_path=files["test"], sample_path=files.get("sample"),
        dictionary_path=files.get("dictionary"), submission_path=str(task_md.parent / "submission.csv"),
        target=cfg_dict["target"], id_col=cfg_dict["id_col"], task_type=cfg_dict["task_type"],
        metric=cfg_dict["metric"], optimization_direction=cfg_dict["optimization_direction"],
        prediction_format=cfg_dict["prediction_format"], submission_columns=list(cfg_dict["submission_columns"]),
        positive_label=cfg_dict.get("positive_label"), task_description=cfg_dict.get("task_description", ""),
        notes=list(cfg_dict.get("notes", [])), source=source,
    )
    if not cfg.task_description:
        cfg.task_description = (f"{cfg.task_type}: предсказать '{cfg.target}', метрика {cfg.metric} "
                                f"({cfg.optimization_direction}).")
    if work_dir is not None:
        save_task_config(cfg, Path(work_dir) / "task_config.json")
    log(f"Постановка: target={cfg.target}, id={cfg.id_col}, type={cfg.task_type}, metric={cfg.metric} "
        f"({cfg.optimization_direction}), format={cfg.prediction_format}, columns={cfg.submission_columns}")
    return cfg


# ----------------------------------------------------------------------------- persistence

def save_task_config(cfg: TaskConfig, path: Path) -> None:
    d = cfg.to_dict()
    pl = d.get("positive_label")
    if isinstance(pl, (np.generic,)):
        d["positive_label"] = pl.item()
    write_json(path, d)


_ALIASES = {
    "target_col": "target", "metric_name": "metric", "direction": "optimization_direction",
}


def load_task_config(path: str | Path) -> dict:
    """Прочитать task_config.json любого из форматов команды и привести к единой схеме (dict)."""
    path = Path(path)
    if path.is_dir():
        path = path / "task_config.json"
    raw = read_json(path, default=None)
    if raw is None:
        raise FileNotFoundError(f"нет {path}")
    d = dict(raw)
    for old, new in _ALIASES.items():
        if old in d and new not in d:
            d[new] = d[old]
    tt = str(d.get("task_type", "")).lower()
    if tt == "classification":
        d["task_type"] = "multiclass_classification"
    if "optimization_direction" not in d and d.get("metric"):
        m = resolve_metric(d["metric"], d.get("task_type"))
        d["optimization_direction"] = m.direction if m else "maximize"
    d.setdefault("target_col", d.get("target"))
    d.setdefault("metric_name", d.get("metric"))
    return d
