"""resource_controller.py — задача 6: контроль времени, LLM-бюджета и вычислений.

Платформа даёт 3 часа на задачу (8 CPU / 32 GB) и убивает процесс без предупреждения.
Контроллер делит бюджет на три части:

    [ исследование .......................... | финальный резерв | страховка ]
      эксперименты, гипотезы, LLM              бленд, постобработка,  запись файлов
                                               refit, финальный сабмит

и отвечает на вопросы оркестратора:
  * можно ли запускать эксперимент с такой оценкой длительности (can_start);
  * какой таймаут дать подпроцессу эксперимента (timeout_for);
  * можно ли ещё звать LLM (llm_allowed) — лимит вызовов и токенов;
  * сколько ресурсов уже потрачено (status_text — короткая сводка для промпта LLM).

Все лимиты переопределяются переменными окружения (удобно для локальных тестов):
    AGENT_TIME_BUDGET_SEC   общий бюджет на задачу, по умолчанию 10200 (2 ч 50 мин — запас до лимита платформы 3 ч)
    AGENT_FINAL_RESERVE_SEC резерв на финальную стадию (по умолчанию 10% бюджета, 60..900 с)
    AGENT_MAX_ITERATIONS    максимум итераций research loop (по умолчанию 40)
    AGENT_N_THREADS         потоки для моделей (по умолчанию — реальный лимит CPU контейнера, см. available_cpus)
    LLM_MAX_CALLS           максимум вызовов LLM на задачу (по умолчанию 120)
    LLM_MAX_TOKENS          максимум токенов (prompt+completion) на задачу (по умолчанию 600000)
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field

from agent_utils import fmt_duration


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def available_cpus() -> int:
    """Реальное число доступных CPU с учётом Docker/cgroup.

    В контейнере с лимитом --cpus=8 os.cpu_count() часто возвращает число ядер всей машины (32, 64...).
    Если отдать столько потоков бустингам, они будут драться за 8 ядер и замедлятся в десятки раз
    (OpenMP крутится в spin-wait). Поэтому берём минимум из affinity и квоты cgroup v1/v2.
    """
    counts = [os.cpu_count() or 1]
    try:
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    try:  # cgroup v2: "<quota> <period>" или "max <period>"
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            counts.append(max(1, int(float(quota) / float(period))))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0 and period > 0:
            counts.append(max(1, quota // period))
    except (OSError, ValueError):
        pass
    return max(1, min(counts))


@dataclass
class ResourceController:
    start_time: float = field(default_factory=time.time)
    budget_sec: float = field(default_factory=lambda: _env_float("AGENT_TIME_BUDGET_SEC", 2 * 3600 + 50 * 60))

    def __post_init__(self) -> None:
        b = self.budget_sec
        self.safety_sec = _clamp(0.02 * b, min(20.0, 0.05 * b), 240)
        self.final_reserve_sec = _env_float("AGENT_FINAL_RESERVE_SEC", _clamp(0.10 * b, min(60.0, 0.15 * b), 900))
        self.max_iterations = int(_env_float("AGENT_MAX_ITERATIONS", 40))
        self.n_threads = int(_env_float("AGENT_N_THREADS", available_cpus()))
        self.llm_max_calls = int(_env_float("LLM_MAX_CALLS", 120))
        self.llm_max_tokens = int(_env_float("LLM_MAX_TOKENS", 600_000))
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_failures = 0
        self.durations: list[tuple[str, float, str, float]] = []  # (family, seconds, status, work_units)
        self.iterations = 0

    # ------------------------------------------------------------------ time
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def remaining(self) -> float:
        """Сколько осталось до момента, когда сабмит обязан быть записан."""
        return self.budget_sec - self.safety_sec - self.elapsed()

    def remaining_research(self) -> float:
        """Сколько осталось на исследование (без финального резерва)."""
        return self.remaining() - self.final_reserve_sec

    def research_deadline_ts(self) -> float:
        return time.time() + max(0.0, self.remaining_research())

    def final_deadline_ts(self) -> float:
        return time.time() + max(0.0, self.remaining())

    # ------------------------------------------------------------------ experiments
    def record_duration(self, family: str, seconds: float, status: str, units: float = 1.0) -> None:
        self.durations.append((family, float(seconds), status, max(float(units), 1e-3)))

    def estimate(self, family: str, units: float = 1.0) -> float:
        """Оценка длительности эксперимента семейства family (registry:<model>, code, blend...).

        units — объём работы относительно прошлых экспериментов (model_zoo.work_units: раунды бустинга
        с учётом early stopping, глубина, число деревьев). Время на единицу работы берётся по семейству.
        """
        if family.startswith(("blend", "postprocess")):
            return 30.0
        same = [d / u for f, d, s, u in self.durations if f == family and s == "success"]
        if same:
            return statistics.median(same) * units * 1.2
        ok = [d for f, d, s, u in self.durations if s == "success" and not f.startswith(("blend", "postprocess"))]
        if ok:
            return statistics.median(ok) * 1.6 * max(1.0, units)
        return max(60.0, 0.04 * self.budget_sec) * max(1.0, units)

    def can_start(self, estimated_sec: float) -> bool:
        rr = self.remaining_research()
        return rr > 20 and rr > estimated_sec * 1.1

    def timeout_for(self, estimated_sec: float, final_stage: bool = False) -> float:
        avail = self.remaining() if final_stage else self.remaining_research()
        floor = 180.0
        if not any(s == "success" for _, _, s, _ in self.durations):
            floor = max(floor, 0.15 * self.budget_sec)  # длительности ещё не измерены — не рубим первый эксперимент
        return max(20.0, min(avail, max(3.0 * estimated_sec, floor)))

    def research_continues(self) -> bool:
        return self.iterations < self.max_iterations and self.remaining_research() > 30

    # ------------------------------------------------------------------ LLM
    def llm_allowed(self, est_tokens: int = 6000) -> bool:
        return (
            self.llm_calls < self.llm_max_calls
            and self.prompt_tokens + self.completion_tokens + est_tokens <= self.llm_max_tokens
            and self.remaining() > 60
        )

    def record_llm(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.llm_calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)

    def llm_timeout(self) -> float:
        return max(20.0, min(240.0, self.remaining() - 30))

    # ------------------------------------------------------------------ reporting
    def status_text(self) -> str:
        n_ok = sum(1 for _, _, s, _ in self.durations if s == "success")
        return (
            f"прошло {fmt_duration(self.elapsed())} из {fmt_duration(self.budget_sec)}; "
            f"на исследование осталось {fmt_duration(max(0, self.remaining_research()))} "
            f"(+{fmt_duration(self.final_reserve_sec)} резерв на финал); "
            f"экспериментов: {len(self.durations)} (успешных {n_ok}); "
            f"LLM: {self.llm_calls}/{self.llm_max_calls} вызовов, "
            f"{(self.prompt_tokens + self.completion_tokens) // 1000}k/{self.llm_max_tokens // 1000}k токенов; "
            f"потоков: {self.n_threads}"
        )

    def to_dict(self) -> dict:
        return {
            "budget_sec": self.budget_sec,
            "elapsed_sec": round(self.elapsed(), 1),
            "final_reserve_sec": self.final_reserve_sec,
            "iterations": self.iterations,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "experiments": [{"family": f, "sec": round(d, 1), "status": s, "units": round(u, 2)}
                            for f, d, s, u in self.durations],
            "n_threads": self.n_threads,
        }
