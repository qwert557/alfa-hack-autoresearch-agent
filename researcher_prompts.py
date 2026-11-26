"""
Модуль для динамической сборки промптов Researcher Agent.
"""
from __future__ import annotations
import json
from typing import Any

def build_system_prompt(task_info: dict) -> str:
    return f"""
Ты — Lead ML Researcher на хакатоне по AutoML. Твоя цель — максимально улучшить метрику {task_info['metric']} ({task_info['optimization_direction']}).

ПРАВИЛА РАБОТЫ:
1. Ты можешь менять ЛЮБОЙ элемент пайплайна: данные, признаки, модель, loss, валидацию, гиперпараметры, постобработку.
2. Используй инструменты (Tools) для уточнения деталей, но не злоупотребляй ими. Как только у тебя достаточно информации для гипотезы — НЕМЕДЛЕННО переходи к финальному выводу.
3. Твой ФИНАЛЬНЫЙ ответ ДОЛЖЕН быть строго валидным JSON-объектом. 
   - НЕ пиши markdown-отчеты с заголовками (## Анализ...).
   - НЕ пиши пояснительный текст до или после JSON.
   - НЕ используй `null` для строковых полей. Используй "baseline" для builds_upon, если это первый эксперимент.

ПРИМЕР ВАЛИДНОГО JSON (структура):
{{
  "hypothesis_id": "hyp_001_baseline",
  "builds_upon": "baseline",
  "statement": "Использование CatBoost с нативной обработкой категорий и StratifiedKFold.",
  "evidence": [{{"source": "EDA", "finding": "Признак proto имеет 133 уникальных значения", "features": ["proto"]}}],
  "mechanism": "CatBoost лучше справляется с высокоразмерными категориями без OHE.",
  "plan": "Обучить CatBoostClassifier с cat_features=['proto'], оценить через 5-кратный StratifiedKFold.",
  "interventions": [{{"axis": "model_architecture", "action": "CatBoostClassifier", "params": {{"iterations": 1000}}}}],
  "experiment_kind": "intervention",
  "expected_effect": "Формирование стабильного baseline с Macro F1 ~0.65.",
  "falsification": "Если Macro F1 < 0.60 или время обучения > 10 минут.",
  "cost_estimate": {{"fit_minutes": 5, "tuning_trials": 0, "extra_memory_risk": "low"}},
  "risks": ["Может не учесть сильные выбросы"],
  "confidence": "high",
  "priority": 1
}}
"""

def build_initial_user_prompt(compact_context: dict, data_dict_summary: str) -> str:
    """Промпт для генерации ПЕРВОЙ гипотезы (Cold Start)."""
    context_json = json.dumps(compact_context, ensure_ascii=False, indent=2)

    return f"""
[ИСХОДНЫЕ ДАННЫЕ (EDA SUMMARY)]
{context_json}

[ОПИСАНИЕ ПРИЗНАКОВ]
{data_dict_summary}

[ИНСТРУКЦИЯ ДЛЯ ПЕРВОГО ЭКСПЕРИМЕНТА]
Это самый первый эксперимент. У тебя нет истории.
Твоя задача — сформулировать СИЛЬНЫЙ baseline или атаковать самую критичную проблему из EDA.
Гипотеза должна включать выбор модели, базового лосса, схемы валидации и (опционально) 
базовую обработку данных. Не ограничивайся только "заполнить пропуски".
Используй инструменты, если нужно уточнить природу данных.
Верни ответ СТРОГО в формате JSON.
"""

def build_iterative_user_prompt(
    compact_context: dict,
    history_summary: str,
    last_reflection: str
) -> str:
    """Промпт для генерации ПОСЛЕДУЮЩИХ гипотез (Итерация N)."""
    context_json = json.dumps(compact_context, ensure_ascii=False, indent=2)

    return f"""
[ИСХОДНЫЕ ДАННЫЕ (EDA SUMMARY)]
{context_json}

[ИСТОРИЯ ЭКСПЕРИМЕНТОВ]
{history_summary}

[РЕФЛЕКСИЯ ПОСЛЕДНЕГО ЗАПУСКА]
"{last_reflection}"

[ИНСТРУКЦИЯ ДЛЯ ИТЕРАТИВНОГО ШАГА]
Это итеративный шаг. Внимательно изучи [РЕФЛЕКСИЮ] и [ИСТОРИЮ].
Твоя новая гипотеза может касаться ЛЮБОГО аспекта (данные, модель, лосс, валидация, 
постобработка, ансамбли). 
Она ДОЛЖНА:
1. Либо развивать успех прошлого шага.
2. Либо исправлять выявленную в рефлексии ошибку (например, переобучение, утечку).
3. НЕ повторять комбинации, которые уже есть в истории.
Если рефлексия указывает на неопределенность, ты можешь предложить ДИАГНОСТИЧЕСКУЮ 
гипотезу (experiment_kind = "diagnostic"), чтобы уменьшить неопределенность.
Верни ответ СТРОГО в формате JSON.
"""

def compress_history_for_prompt(experiment_history: list[dict], max_entries: int = 3,
                                greater_is_better: bool = True) -> str:
    """Сжимает историю экспериментов в короткий текст для экономии токенов."""
    if not experiment_history:
        return "История пуста (это первый эксперимент)."

    # Берем лучший эксперимент и последние неудачные
    scored = [x for x in experiment_history if isinstance(x.get("local_cv_score"), (int, float))]
    sorted_history = sorted(scored, key=lambda x: x["local_cv_score"], reverse=greater_is_better)
    top_best = sorted_history[0] if sorted_history else experiment_history[-1]
    recent = experiment_history[-(max_entries-1):] if len(experiment_history) > 1 else []

    selected = [top_best] + [exp for exp in recent if exp.get('hypothesis_id') != top_best.get('hypothesis_id')]
    selected = selected[:max_entries]

    lines = []
    for i, exp in enumerate(selected, 1):
        status = "Успех" if exp.get("status") == "success" else "Провал/Ошибка"
        reflection = exp.get("reflection") or "Нет рефлексии"
        lines.append(
            f"{i}. [{status}] ID: {exp.get('hypothesis_id')}. "
            f"Метрика: {exp.get('local_cv_score', 'N/A')}. "
            f"Суть: {(exp.get('statement') or '')[:100]}... "
            f"Вывод: {reflection}"
        )
    return "\n".join(lines)