"""llm_client.py — обёртка над OpenAI-совместимым API (Qwen на платформе).

Читает те же переменные окружения, что и baseline.py: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL.
Если OPENAI_MODEL не задан (или AGENT_DISABLE_LLM=1), клиент выключен и агент работает на
детерминированных fallback-ах — сабмит будет в любом случае.

Особенности:
  * бэкенд — пакет openai, если он установлен (он есть в requirements.txt), иначе urllib из stdlib;
  * ретраи с экспоненциальной паузой, таймаут на вызов от ResourceController;
  * «предохранитель»: после 3 подряд неудачных вызовов LLM отключается до конца запуска;
  * учёт токенов (usage) в ResourceController, журнал всех вызовов в llm_log.jsonl;
  * <think>-блоки Qwen3 вырезаются; LLM_NO_THINK=1 (по умолчанию) добавляет «/no_think» к запросу,
    чтобы не тратить токены на рассуждения (для Qwen3 это штатный переключатель).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from agent_utils import extract_json, log, short, strip_think, to_jsonable
from resource_controller import ResourceController


class LLMClient:
    def __init__(self, rc: ResourceController, log_path: str | Path | None = None):
        self.rc = rc
        self.model = os.environ.get("OPENAI_MODEL", "").strip()
        self.base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.no_think = os.environ.get("LLM_NO_THINK", "1") == "1"
        self.enabled = bool(self.model) and os.environ.get("AGENT_DISABLE_LLM", "0") != "1"
        self.log_path = Path(log_path) if log_path else None
        self.failures_in_row = 0
        self._client = None
        self.backend = "disabled"
        if self.enabled:
            backend = os.environ.get("LLM_BACKEND", "auto")
            if backend in ("auto", "openai"):
                try:
                    from openai import OpenAI  # noqa: WPS433

                    self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
                    self.backend = "openai"
                except Exception:  # noqa: BLE001
                    self.backend = "urllib"
            else:
                self.backend = "urllib"
        log(f"LLM: {'включён' if self.enabled else 'выключен'} (backend={self.backend}, model={self.model or '—'})")

    # ------------------------------------------------------------------ low level
    def _call_openai(self, messages: list[dict], max_tokens: int, temperature: float, timeout: float):
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens,
            temperature=temperature, timeout=timeout,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", 0) if usage else 0
        ct = getattr(usage, "completion_tokens", 0) if usage else 0
        return text, pt, ct

    def _call_urllib(self, messages: list[dict], max_tokens: int, temperature: float, timeout: float):
        body = json.dumps({
            "model": self.model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
            data = json.loads(r.read().decode("utf-8"))
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return text, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    def _journal(self, stage: str, messages: list[dict], text: str | None, pt: int, ct: int, err: str | None, sec: float):
        if not self.log_path:
            return
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "sec": round(sec, 1),
            "prompt_chars": sum(len(m.get("content", "")) for m in messages),
            "prompt_tokens": pt, "completion_tokens": ct, "error": err,
            "last_user": short(messages[-1].get("content", ""), 400),
            "response": text,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(to_jsonable(rec), ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------ public
    def available(self, est_tokens: int = 6000) -> bool:
        return self.enabled and self.rc.llm_allowed(est_tokens)

    def chat(self, messages: Sequence[dict], max_tokens: int = 2000, temperature: float = 0.3,
             stage: str = "") -> str | None:
        """Один запрос к LLM. Возвращает текст без <think> или None (LLM недоступна/упала)."""
        est = sum(len(m.get("content", "")) for m in messages) // 3 + max_tokens
        if not self.available(est):
            return None
        msgs = [dict(m) for m in messages]
        if self.no_think and msgs and msgs[-1]["role"] == "user" and "/no_think" not in msgs[-1]["content"]:
            msgs[-1]["content"] = msgs[-1]["content"] + "\n/no_think"
        for attempt in range(3):
            t0 = time.time()
            try:
                timeout = self.rc.llm_timeout()
                call = self._call_openai if self.backend == "openai" else self._call_urllib
                text, pt, ct = call(msgs, max_tokens, temperature, timeout)
                self.rc.record_llm(pt or est - max_tokens, ct or len(text) // 3)
                self._journal(stage, msgs, text, pt, ct, None, time.time() - t0)
                self.failures_in_row = 0
                return strip_think(text)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {short(e, 300)}"
                self._journal(stage, msgs, None, 0, 0, err, time.time() - t0)
                log(f"LLM [{stage}] ошибка (попытка {attempt + 1}/3): {err}")
                if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403, 404):
                    break
                if self.rc.remaining() < 120:
                    break
                time.sleep(2 * (2 ** attempt))
        self.failures_in_row += 1
        self.rc.llm_failures += 1
        if self.failures_in_row >= 3:
            log("LLM: 3 неудачных запроса подряд — отключаю LLM до конца запуска, работаю на fallback-ах")
            self.enabled = False
        return None

    def ask_json(self, system: str, user: str, max_tokens: int = 2000, temperature: float = 0.3,
                 stage: str = "", required_keys: Sequence[str] = ()) -> tuple[dict | None, str | None]:
        """Запрос, ответ на который должен содержать JSON. Одна попытка починки при ошибке разбора.

        Возвращает (json_dict | None, сырой_текст | None).
        """
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.chat(messages, max_tokens=max_tokens, temperature=temperature, stage=stage)
        if text is None:
            return None, None
        for attempt in range(2):
            try:
                obj = extract_json(text)
                missing = [k for k in required_keys if k not in obj]
                if missing:
                    raise ValueError(f"нет обязательных ключей: {missing}")
                return obj, text
            except ValueError as e:
                if attempt == 1:
                    log(f"LLM [{stage}]: ответ не разобран ({e})")
                    return None, text
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Ответ не удалось разобрать: {e}. Верни ответ заново строго в "
                                                f"требуемом формате: валидный JSON-объект (и, если требовалось, "
                                                f"блок ```python после него), без пояснений."},
                ]
                text = self.chat(messages, max_tokens=max_tokens, temperature=0.1, stage=stage + ":repair")
                if text is None:
                    return None, None
        return None, text
