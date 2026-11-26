"""
Модуль инструментов (Tools) для Researcher Agent.
Оборачивает методы DataProfiler в LangChain Tools для получения
детальной информации по конкретным признакам в процессе рассуждений.
"""
from __future__ import annotations
import json
from typing import Any
from langchain_core.tools import tool
from data_profiler import DataProfiler

def create_researcher_tools(profiler: DataProfiler) -> list:
    """
    Фабрика инструментов для Researcher Agent.

    Parameters
    ----------
    profiler : DataProfiler
        Инициализированный объект профайлера.

    Returns
    -------
    list[BaseTool]
        Список инструментов для передачи в LLM через bind_tools().
    """

    @tool
    def get_column_full_profile(feature_name: str) -> str:
        """
        Возвращает детальный статистический профиль конкретного признака
        (квантили, IQR-выбросы, редкие категории).
        Вызывай, если нужно понять природу пропусков или выбросов.
        """
        try:
            result = profiler.inspect_column(feature_name, detail_level="full")
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return f"Ошибка при получении профиля: {str(e)}"

    @tool
    def get_feature_target_association(feature_name: str) -> str:
        """
        Возвращает силу и тип связи признака с целевой переменной.
        Вызывай, чтобы понять, стоит ли генерировать фичи на его основе.
        """
        try:
            result = profiler.analyze_feature_target(feature_name)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return f"Ошибка при анализе связи с таргетом: {str(e)}"

    @tool
    def search_data_dictionary(query: str) -> str:
        """
        Ищет признаки в Data Dictionary по смыслу, названию или описанию.
        Вызывай, если не понятна бизнес-логика признака (например, что значит -1).
        """
        try:
            result = profiler.search_data_dictionary(query)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return f"Ошибка поиска в словаре: {str(e)}"

    @tool
    def get_train_test_shift_details(feature_name: str) -> str:
        """
        Возвращает детали сдвига распределения между train и test.
        Вызывай, если в контексте указан сдвиг, и нужно понять его масштаб.
        """
        try:
            result = profiler.analyze_train_test_shift([feature_name])
            return json.dumps(result.get(feature_name, {}), ensure_ascii=False, default=str)
        except Exception as e:
            return f"Ошибка при анализе сдвига: {str(e)}"

    return [
        get_column_full_profile,
        get_feature_target_association,
        search_data_dictionary,
        get_train_test_shift_details
    ]