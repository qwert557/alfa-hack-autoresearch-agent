from __future__ import annotations
import json
import re
from typing import Annotated, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ValidationError, Field

from researcher_tools import create_researcher_tools
from researcher_prompts import (
    build_system_prompt,
    build_initial_user_prompt,
    build_iterative_user_prompt,
    compress_history_for_prompt
)


# ============================================================
# Pydantic модели (Сделаны максимально устойчивыми)
# ============================================================
class EvidenceItem(BaseModel):
    source: str = "EDA"
    finding: str
    features: list[str] = Field(default_factory=list)


class InterventionItem(BaseModel):
    axis: str
    action: str
    params: dict[str, Any] = Field(default_factory=dict)


class CostEstimate(BaseModel):
    fit_minutes: int = 5
    tuning_trials: int = 0
    extra_memory_risk: str = "low"


class HypothesisSchema(BaseModel):
    hypothesis_id: str
    builds_upon: str = "baseline"  # Дефолт, если LLM забудет или напишет null
    statement: str
    evidence: list[EvidenceItem]
    mechanism: str
    plan: str
    interventions: list[InterventionItem] = Field(default_factory=list)
    experiment_kind: str = "intervention"
    expected_effect: str
    falsification: str
    cost_estimate: CostEstimate
    risks: list[str] = Field(default_factory=list)
    confidence: str = "medium"
    priority: int = 1


# ============================================================
# Стейт графа LangGraph
# ============================================================
class ResearcherState(TypedDict):
    messages: Annotated[list, add_messages]
    compact_context: dict
    experiment_history: list
    last_reflection: str
    final_hypothesis: dict | None
    parse_error: str | None


# ============================================================
# Главный класс агента
# ============================================================
class HypothesisResearcher:
    def __init__(self, llm: ChatOpenAI, profiler: Any, task_info: dict):
        self.task_info = task_info
        self.tools = create_researcher_tools(profiler)
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ResearcherState)
        workflow.add_node("call_model", self._call_model_node)
        workflow.add_node("call_tools", self.tool_node)
        workflow.add_node("parse_and_validate", self._parse_and_validate_node)

        workflow.set_entry_point("call_model")
        workflow.add_conditional_edges(
            "call_model",
            self._should_continue,
            {"continue": "call_tools", "finish": "parse_and_validate"},
        )
        workflow.add_edge("call_tools", "call_model")
        workflow.add_conditional_edges(
            "parse_and_validate",
            self._check_parse_result,
            {"success": END, "retry": "call_model"}
        )
        return workflow.compile()

    def _call_model_node(self, state: ResearcherState) -> dict:
        messages = list(state["messages"])
        extra = []

        if state.get("parse_error"):
            error_msg = (
                "КРИТИЧЕСКАЯ ОШИБКА: Твой JSON невалиден или обрезан. "
                f"Детали: {state['parse_error']}\n\n"
                "ПРАВИЛА ИСПРАВЛЕНИЯ:\n"
                "1. Убедись, что JSON начинается с '{' и заканчивается '}'.\n"
                "2. НЕ используй null для строковых полей. Используй 'baseline' для builds_upon.\n"
                "3. Включи ВСЕ обязательные поля: hypothesis_id, statement, evidence, mechanism, plan, expected_effect, falsification, cost_estimate, confidence, priority.\n"
                "4. Не обрезай текст. Выдай полный JSON."
            )
            extra.append(HumanMessage(content=error_msg))
            print(f"\n[DEBUG] Отправлена строгая ошибка валидации в LLM.")

        response = self.llm_with_tools.invoke(messages + extra)

        if hasattr(response, "tool_calls") and response.tool_calls:
            print(f"[DEBUG] LLM вызвала инструменты: {[tc['name'] for tc in response.tool_calls]}")
        else:
            print(f"[DEBUG] LLM выдала финальный текст (длина: {len(response.content)} симв.)")

        return {"messages": extra + [response], "parse_error": None}

    def _should_continue(self, state: ResearcherState) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
        return "finish"

    def _parse_and_validate_node(self, state: ResearcherState) -> dict:
        raw_text = state["messages"][-1].content
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        raw_text = re.sub(r"<think>[\s\S]*?(</think>|$)", "", raw_text).strip()

        # ИСПРАВЛЕННЫЙ ПАРСЕР: берем ВЕСЬ блок между ```json и ```, а не ищем первую }
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text, re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            print("[DEBUG] Найден блок ```json. Извлекаю содержимое целиком.")
        else:
            # Fallback: ищем первую { и последнюю }
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = raw_text[start:end + 1]
                print("[DEBUG] Блок ```json не найден, извлечено между первой { и последней }.")
            else:
                json_str = raw_text
                print("[DEBUG] Фигурные скобки не найдены, пытаюсь распарсить как есть.")

        try:
            parsed_dict = json.loads(json_str)
            validated_hypothesis = HypothesisSchema(**parsed_dict).model_dump()
            print("[DEBUG] ✅ Валидация Pydantic УСПЕШНА!")
            return {"final_hypothesis": validated_hypothesis, "parse_error": None}
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[DEBUG] ❌ Валидация ПРОВАЛЕНА. Первые 200 символов извлеченной строки:\n{json_str[:200]}")
            return {"final_hypothesis": None, "parse_error": str(e)}

    def _check_parse_result(self, state: ResearcherState) -> str:
        if state.get("final_hypothesis"):
            return "success"
        return "retry"

    def generate_hypothesis(self, compact_context: dict, experiment_history: list, last_reflection: str,
                            greater_is_better: bool = True, extra_instructions: str = "") -> dict:
        if not experiment_history:
            data_dict_summary = json.dumps(compact_context.get("data_dictionary", []), ensure_ascii=False)
            user_prompt = build_initial_user_prompt(compact_context, data_dict_summary)
        else:
            history_summary = compress_history_for_prompt(experiment_history, greater_is_better=greater_is_better)
            user_prompt = build_iterative_user_prompt(compact_context, history_summary, last_reflection)
        if extra_instructions:
            user_prompt += "\n" + extra_instructions

        system_prompt = build_system_prompt(self.task_info)
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        initial_state = {
            "messages": initial_messages,
            "compact_context": compact_context,
            "experiment_history": experiment_history,
            "last_reflection": last_reflection,
            "final_hypothesis": None,
            "parse_error": None
        }

        print("[DEBUG] Запуск графа LangGraph...")
        # Увеличим лимит до 12, чтобы дать 2-3 шанса на исправление после наших строгих подсказок
        final_state = self.graph.invoke(initial_state, {"recursion_limit": 12})

        if final_state.get("final_hypothesis"):
            return final_state["final_hypothesis"]

        raise RuntimeError("Researcher Agent не смог сгенерировать валидную гипотезу.")