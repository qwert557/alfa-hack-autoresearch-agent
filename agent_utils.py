"""agent_utils.py — общие утилиты агента.

Логирование с таймстемпом, атомарная запись JSON, приведение numpy-типов к JSON,
извлечение JSON и Python-кода из ответов LLM (в т.ч. с <think>-блоками Qwen3),
markdown-таблицы без зависимости от tabulate (его нет в requirements.txt).
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_START = time.time()
_LOG_FILE: Path | None = None


def set_log_file(path: str | Path) -> None:
    """Дублировать лог агента в файл (кроме stdout)."""
    global _LOG_FILE
    _LOG_FILE = Path(path)
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')} +{time.time() - _START:7.1f}s] {msg}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


# ----------------------------------------------------------------------------- JSON

def to_jsonable(obj: Any) -> Any:
    """Рекурсивно привести numpy/pandas/Path к JSON-совместимым типам. NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return to_jsonable(obj.item())
        except (ValueError, TypeError):
            pass
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    """Атомарная запись: сначала во временный файл, потом os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- LLM output parsing

_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.S)


def strip_think(text: str | None) -> str:
    """Убрать рассуждения Qwen3 (<think>...</think>), включая незакрытый блок."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.I)[-1]
    if text.lstrip().lower().startswith("<think>"):
        # незакрытый think — ответа нет
        return ""
    return text.strip()


def _balanced_objects(text: str) -> Iterable[str]:
    """Найти подстроки вида {...} со сбалансированными скобками с учётом строк."""
    depth = 0
    start = None
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:i + 1]
                start = None


def _relaxed_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # убрать // комментарии в начале строк и висячие запятые
    cleaned = re.sub(r"(?m)^\s*//.*$", "", s)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    cleaned = cleaned.replace("“", '"').replace("”", '"')
    return json.loads(cleaned)


def extract_json(text: str | None) -> dict:
    """Достать первый валидный JSON-объект из ответа LLM. Бросает ValueError, если не нашёл."""
    text = strip_think(text)
    if not text:
        raise ValueError("пустой ответ")
    candidates: list[str] = []
    for lang, body in _FENCE_RE.findall(text):
        if lang.lower() in ("json", "jsonc", "") and body.strip().startswith("{"):
            candidates.append(body.strip())
    candidates.extend(_balanced_objects(text))
    last_err: Exception | None = None
    for cand in candidates:
        try:
            obj = _relaxed_json_loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"не удалось извлечь JSON: {last_err}")


def extract_code(text: str | None, must_contain: str = "def fit_predict") -> str | None:
    """Достать блок Python-кода (предпочтительно содержащий must_contain)."""
    text = strip_think(text)
    blocks = [body for lang, body in _FENCE_RE.findall(text) if lang.lower() in ("python", "py", "")]
    for body in blocks:
        if must_contain in body:
            return body.strip() + "\n"
    if must_contain in text and not blocks:
        idx = text.find("import ")
        idx = idx if 0 <= idx < text.find(must_contain) else text.find(must_contain)
        return text[idx:].strip() + "\n"
    return None


# ----------------------------------------------------------------------------- formatting

def short(text: Any, limit: int = 300) -> str:
    s = str(text) if text is not None else ""
    s = s.replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def tail(text: str, limit: int = 3000) -> str:
    return text if len(text) <= limit else "…" + text[-limit:]


def fmt_num(x: Any, digits: int = 5) -> str:
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(xf):
        return "—"
    if abs(xf) >= 1e5 or (0 < abs(xf) < 1e-3):
        return f"{xf:.3e}"
    return f"{xf:.{digits}g}"


def fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}ч {m:02d}м"
    if m:
        return f"{m}м {s:02d}с"
    return f"{s}с"


def md_table(rows: list[dict], columns: list[str], headers: list[str] | None = None) -> str:
    """Markdown-таблица без tabulate."""
    headers = headers or columns
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c)
            cells.append(fmt_num(v) if isinstance(v, (float, np.floating)) else short(v, 120).replace("|", "/"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
