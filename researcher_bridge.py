"""researcher_bridge.py — связка Researcher Agent (части 1–3) с остальным пайплайном.

Researcher Agent (researcher_agent.py, LangGraph + инструменты над DataProfiler) думает над данными и
выдаёт гипотезу в формате команды (statement / evidence / interventions / ...). Мост:
  * один раз строит профиль данных (profile_layers) на подвыборке train, пока сырые данные ещё в памяти;
  * создаёт ChatOpenAI с теми же OPENAI_* переменными, что и llm_client, и подключает к нему
    контроллер ресурсов: каждый вызов учитывается в лимитах, а при нехватке времени/вызовов прерывается;
  * переводит историю экспериментов в формат, который ожидают промпты researcher_prompts;
  * при любой ошибке (нет langchain, модель не умеет tools, невалидный JSON) возвращает None —
    HypothesisEngine тогда работает на своём LLM-генераторе и fallback-лестнице.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path

import pandas as pd

from agent_utils import log, short, write_json

MAX_PROFILE_ROWS = 100_000


class _BudgetExceeded(RuntimeError):
    pass


def _dictionary_frame(path) -> pd.DataFrame | None:
    if not path:
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


class ResearcherBridge:
    def __init__(self, cfg, metric, rc, work_dir: Path, train: pd.DataFrame, test: pd.DataFrame):
        self.cfg, self.metric, self.rc, self.work_dir = cfg, metric, rc, Path(work_dir)
        self.enabled = False
        self.failures = 0
        self.researcher = None
        self.compact_context: dict = {}
        if os.environ.get("AGENT_DISABLE_RESEARCHER", "0") == "1" or os.environ.get("AGENT_DISABLE_LLM", "0") == "1":
            return
        if not os.environ.get("OPENAI_MODEL"):
            return
        t0 = time.time()
        try:
            self._build(train, test)
            self.enabled = True
            log(f"Researcher Agent готов: профиль данных за {time.time() - t0:.1f}с")
        except Exception as e:  # noqa: BLE001
            log(f"Researcher Agent отключён: {type(e).__name__}: {short(e, 300)}")
            write_json(self.work_dir / "researcher_error.json", {"error": traceback.format_exc()})

    # ------------------------------------------------------------------ setup
    def _build(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_openai import ChatOpenAI

        from data_profiler import DataProfiler
        from dataset_intake import build_intake_contract
        from profile_layers import build_profile_layers, save_profile_layers
        from researcher_agent import HypothesisResearcher

        cfg = self.cfg
        dictionary = _dictionary_frame(getattr(cfg, "dictionary_path", None))
        tr = train.sample(MAX_PROFILE_ROWS, random_state=0) if len(train) > MAX_PROFILE_ROWS else train
        te = test.sample(MAX_PROFILE_ROWS, random_state=0) if len(test) > MAX_PROFILE_ROWS else test
        drop_tr = [c for c in tr.columns if c == cfg.id_col]
        drop_te = [c for c in te.columns if c == cfg.id_col or c not in tr.columns]
        tr, te = tr.drop(columns=drop_tr), te.drop(columns=drop_te)
        direction = "maximize" if self.metric.greater_is_better else "minimize"

        try:
            req = Path(__file__).resolve().parent / "requirements.txt"
            contract = build_intake_contract(tr, te, cfg.target, cfg.task_type, self.metric.name, direction,
                                             dictionary, req if req.exists() else None)
            write_json(self.work_dir / "intake_contract.json", contract)
        except Exception as e:  # noqa: BLE001
            log(f"intake contract: {short(e, 200)}")

        layers = None
        for budget in (2_000, 4_000, 8_000):
            try:
                layers = build_profile_layers(tr, te, cfg.target, cfg.task_type, self.metric.name, direction,
                                              dictionary, compact_token_budget=budget)
                break
            except ValueError as e:
                if "token_budget" not in str(e):
                    raise
        if layers is None:
            raise RuntimeError("компактный профиль не уложился в 8000 токенов")
        save_profile_layers(layers, self.work_dir / "profile")
        self.compact_context = layers["compact_context"]

        rc = self.rc

        class _Budget(BaseCallbackHandler):
            raise_error = True

            def on_chat_model_start(self, serialized, messages, **kwargs):
                if not rc.llm_allowed(8000) or rc.remaining_research() < 120:
                    raise _BudgetExceeded("лимит LLM или времени исчерпан")

            def on_llm_end(self, response, **kwargs):
                usage = (response.llm_output or {}).get("token_usage") or {}
                rc.record_llm(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

        llm = ChatOpenAI(
            model=os.environ["OPENAI_MODEL"],
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
            temperature=0.4, max_tokens=2500, timeout=min(120.0, rc.llm_timeout()), max_retries=1,
            callbacks=[_Budget()],
        )
        profiler = DataProfiler(tr, cfg.target, cfg.task_type, te, dictionary)
        task_info = {"metric": self.metric.name, "optimization_direction": direction}
        self.researcher = HypothesisResearcher(llm, profiler, task_info)

    # ------------------------------------------------------------------ use
    def available(self) -> bool:
        return (self.enabled and self.researcher is not None and self.failures < 2
                and self.rc.llm_allowed(15000) and self.rc.remaining_research() > 300)

    def propose(self, history: list[dict], last_reflection: str, extra: str = "") -> dict | None:
        if not self.available():
            return None
        hist = [{"hypothesis_id": h.get("exp_id"), "statement": h.get("hypothesis_summary"),
                 "local_cv_score": h.get("local_cv_score"), "status": h.get("status"),
                 "reflection": h.get("reflection") or h.get("error_message")} for h in history]
        t0 = time.time()
        try:
            h = self.researcher.generate_hypothesis(
                self.compact_context, hist, last_reflection or "", greater_is_better=self.metric.greater_is_better,
                extra_instructions=(extra + "\n/no_think").strip())
            log(f"Researcher Agent: гипотеза за {time.time() - t0:.0f}с — {short(h.get('statement'), 160)}")
            self.failures = 0
            return h
        except Exception as e:  # noqa: BLE001
            self.failures += 1
            log(f"Researcher Agent не дал гипотезу ({type(e).__name__}: {short(e, 200)}) за {time.time() - t0:.0f}с")
            return None
