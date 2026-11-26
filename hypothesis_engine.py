"""hypothesis_engine.py — задача 3: формирование гипотез («мыслитель» + «критик»).

Мыслитель — LLM: получает постановку, EDA, схему валидации, историю с рефлексиями, анализ ошибок
и остаток ресурсов, и предлагает ОДНУ гипотезу в формате hypothesis_001.json + implementation:
    registry    — модель реестра с любыми параметрами (loss, глубина, регуляризация, балансировка...)
    code        — своя функция fit_predict (признаки, нестандартные модели) — LLM пишет код
    blend       — ансамбль уже посчитанных экспериментов
    postprocess — веса классов на OOF для метрик по меткам

Критик — детерминированные проверки до запуска (дешевле, чем тратить на это ещё вызов LLM):
    модель доступна, kind известен, код компилируется и содержит fit_predict, нет запрещённых операций
    (файлы, сеть, subprocess), гипотеза не повторяет уже проверенную, укладывается в оставшееся время.
    Замечания критика отправляются LLM один раз; если и вторая попытка плохая — берём fallback.

Fallback-лестница — заранее упорядоченный список сильных гипотез, зависящий от данных (дисбаланс,
асимметрия таргета, категориальные признаки, доступные библиотеки). Работает, когда LLM недоступна
или исчерпан бюджет токенов, и гарантирует осмысленное исследование без LLM.
"""

from __future__ import annotations

import json
import re

import model_zoo
import prompts
from agent_utils import extract_code, fmt_duration, fmt_num, log, short
from ml_analysis import history_table
from ml_experiment import (ExperimentStore, Workspace, family_of, normalize_hypothesis, pick_blend_candidates,
                           spec_signature, work_units_of)

KINDS = ("registry", "code", "blend", "postprocess")
FORBIDDEN = [
    (r"\bopen\s*\(", "работа с файлами"), (r"read_csv|read_parquet|to_csv|to_parquet|np\.save|np\.load", "чтение/запись файлов"),
    (r"\bsubprocess\b|os\.system|os\.popen|os\.remove|shutil", "системные вызовы"),
    (r"\brequests\b|urllib|socket|http\.client", "сеть"), (r"\bexit\s*\(|sys\.exit", "завершение процесса"),
]
BALANCE_METRICS = ("f1_macro", "balanced_accuracy", "recall_macro", "precision_macro", "mcc", "f1", "recall",
                   "qwk", "cohen_kappa", "f1_weighted")


def _h(statement: str, impl: dict, mechanism: str = "", expected: str = "") -> dict:
    return {"hypothesis_id": "", "builds_upon": "best", "statement": statement, "summary": statement,
            "mechanism": mechanism, "implementation": impl, "expected_effect": expected,
            "falsification": "метрика OOF не лучше текущей лучшей", "confidence": "medium"}


class HypothesisEngine:
    def __init__(self, ws: Workspace, store: ExperimentStore, llm, profile: dict, eda_summary: str, researcher=None):
        self.ws, self.store, self.llm, self.profile, self.eda_summary = ws, store, llm, profile, eda_summary
        self.researcher = researcher
        self.last_reflection: str = ""
        self.models = model_zoo.available_models()
        self.gbdt = model_zoo.preferred_gbdt_order()
        self.ladder = self._build_ladder()
        self.last_analysis: str = ""
        self.llm_failures = 0

    # ------------------------------------------------------------------ fallback
    def _needs_balance(self) -> bool:
        if not self.ws.cfg.is_classification:
            return False
        t = self.profile.get("target", {})
        return self.ws.metric.name in BALANCE_METRICS and float(t.get("imbalance_ratio", 1)) >= 5

    def quick_baseline(self) -> dict:
        main = self.gbdt[0]
        return _h(f"Быстрый бейзлайн: {main} с ускоренными параметрами — опорная точка и первый настоящий сабмит",
                  {"kind": "registry", "model": main, "params": dict(model_zoo.FAST[main]), "features": {}},
                  "градиентный бустинг — сильный дефолт для табличных данных", "заметно лучше константы")

    def _build_ladder(self) -> list[dict]:
        clf = self.ws.cfg.is_classification
        bal = self._needs_balance()
        main = self.gbdt[0]
        second = self.gbdt[1] if len(self.gbdt) > 1 else "extra_trees"
        t = self.profile.get("target", {})
        has_cats = bool(self.ws.meta.get("cat_features"))
        items = [_h(f"{main} с полными параметрами (больше итераций, early stopping)" + (" и балансировкой классов" if bal else ""),
                    {"kind": "registry", "model": main, "params": {"class_balance": True} if bal else {}, "features": {}},
                    "больше итераций и меньший шаг обучения" + ("; веса классов поднимают вклад редких классов в macro-метрику" if bal else ""))]
        if bal:
            items.append(_h(f"{main} без балансировки классов — проверить, не ухудшает ли она частые классы",
                            {"kind": "registry", "model": main, "params": {}, "features": {}}))
        items.append(_h(f"{second} — другая модель для разнообразия и будущего ансамбля",
                        {"kind": "registry", "model": second, "params": {"class_balance": True} if bal else {}, "features": {}}))
        if not clf and float(t.get("skew", 0)) > 1.0 and float(t.get("share_negative", 1)) == 0:
            items.append(_h(f"{main} на log1p(таргет): асимметрия {t.get('skew')} — модель лучше учит основную массу",
                            {"kind": "registry", "model": main, "params": {"target_transform": "log1p"}, "features": {}},
                            "логарифм снижает влияние хвоста на обучение", "лучше для MAE/RMSLE, для RMSE — не факт"))
        if has_cats:
            items.append(_h(f"{main} + частотное кодирование категорий",
                            {"kind": "registry", "model": main, "params": {"class_balance": True} if bal else {},
                             "features": {"frequency_encoding": True}}, "частота категории — устойчивый признак"))
        deeper = {"catboost": {"depth": 8, "learning_rate": 0.04, "l2_leaf_reg": 5},
                  "lightgbm": {"num_leaves": 127, "learning_rate": 0.03, "min_child_samples": 40},
                  "xgboost": {"max_depth": 9, "learning_rate": 0.03, "min_child_weight": 3},
                  "hist_gb": {"max_leaf_nodes": 127, "learning_rate": 0.04, "l2_regularization": 1.0}}[main]
        if bal:
            deeper = {**deeper, "class_balance": True}
        items.append(_h(f"{main} глубже и с меньшим шагом", {"kind": "registry", "model": main, "params": deeper, "features": {}},
                        "больше ёмкость модели для сложных взаимодействий"))
        for m in self.gbdt[2:]:
            items.append(_h(f"{m} — ещё одна модель для ансамбля",
                            {"kind": "registry", "model": m, "params": {"class_balance": True} if bal else {}, "features": {}}))
        items.append(_h("ExtraTrees — непохожая модель для ансамбля",
                        {"kind": "registry", "model": "extra_trees", "params": {"class_balance": True} if bal else {}, "features": {}}))
        if self.profile.get("n_features", 0) <= 300:
            items.append(_h("Линейная модель с one-hot — максимально другая модель для ансамбля",
                            {"kind": "registry", "model": "linear", "params": {"class_balance": True} if bal else {}, "features": {}}))
        return [h for h in items if h["implementation"].get("model") in self.models]

    def _automatic(self) -> dict | None:
        """Бленд и постобработка, когда есть что смешивать."""
        cands = pick_blend_candidates(self.store)
        if len(cands) >= 2:
            h = _h(f"Бленд лучших экспериментов {cands}", {"kind": "blend", "exp_ids": cands},
                   "усреднение непохожих моделей снижает дисперсию ошибки")
            if spec_signature(h["implementation"]) not in self.store.signatures():
                return h
        best = self.store.best()
        if (best and self.ws.cfg.is_classification and self.ws.metric.needs == "label"
                and best.get("kind") != "postprocess"):
            h = _h(f"Подбор весов классов на OOF для {self.ws.metric.name} поверх {best['exp_id']}",
                   {"kind": "postprocess", "base_exp_id": best["exp_id"]},
                   "argmax вероятностей не оптимален для macro-метрик; веса классов сдвигают пороги")
            if spec_signature(h["implementation"]) not in self.store.signatures():
                return h
        return None

    def _next_fallback(self) -> dict | None:
        tried = self.store.signatures()
        n_ok = len([h for h in self.store.successes() if h.get("kind") in ("registry", "code")])
        last_kind = self.store.history[-1].get("kind") if self.store.history else None
        if n_ok >= 3 and last_kind not in ("blend", "postprocess"):
            auto = self._automatic()
            if auto:
                return auto
        for h in self.ladder + self._seed_bagging():
            impl = normalize_hypothesis(h)["implementation"]
            if spec_signature(impl) in tried:
                continue
            if not self.ws.rc.can_start(self.ws.rc.estimate(family_of(impl), work_units_of(impl))):
                continue
            return h
        return self._automatic()

    def _seed_bagging(self, n: int = 3) -> list[dict]:
        """Когда лестница кончилась: лучшая registry-конфигурация с другими сидами — материал для бленда."""
        regs = [h for h in self.store.successes() if h.get("kind") == "registry"]
        if not regs:
            return []
        best = max(regs, key=lambda h: h["local_cv_score"]) if self.store.metric.greater_is_better else \
            min(regs, key=lambda h: h["local_cv_score"])
        impl = (best.get("hypothesis") or {}).get("implementation") or {}
        out = []
        for k in range(1, n + 1):
            params = {**(impl.get("params") or {}), "seed_offset": 1000 * k}
            out.append(_h(f"Бэггинг: {best['exp_id']} ({impl.get('model')}) с другим сидом #{k} — для бленда",
                          {"kind": "registry", "model": impl.get("model"), "params": params,
                           "features": dict(impl.get("features") or {})},
                          "усреднение моделей с разными сидами снижает дисперсию"))
        return out

    # ------------------------------------------------------------------ critic
    def critic(self, h: dict) -> list[str]:
        issues = []
        impl = h.get("implementation") or {}
        kind = impl.get("kind")
        if kind not in KINDS:
            return [f"неизвестный implementation.kind={kind!r}; допустимо: {KINDS}"]
        if kind == "registry":
            if impl.get("model") not in self.models:
                issues.append(f"модель {impl.get('model')!r} недоступна; доступны: {list(self.models)}")
            if not isinstance(impl.get("params", {}), dict):
                issues.append("params должен быть объектом")
            feats = impl.get("features") or {}
            if feats.get("drop_columns"):  # несуществующие колонки (например, id) молча убираем, это не ошибка гипотезы
                known = set(self.ws.meta["cat_features"]) | set(self.ws.meta["num_features"])
                feats["drop_columns"] = [c for c in feats["drop_columns"] if c in known]
                impl["features"] = feats
        elif kind == "code":
            code = impl.get("code") or ""
            if "def fit_predict" not in code:
                issues.append("в коде нет def fit_predict(X_train, y_train, X_val, X_test, ctx)")
            else:
                try:
                    compile(code, "<llm_code>", "exec")
                except SyntaxError as e:
                    issues.append(f"синтаксическая ошибка в коде: {e}")
            for pat, why in FORBIDDEN:
                if re.search(pat, code):
                    issues.append(f"в коде запрещено: {why} ({pat})")
        elif kind == "blend":
            ids = impl.get("exp_ids")
            if ids != "auto":
                ok = [i for i in ids or [] if (r := self.store.get(i)) and r.get("status") == "success" and r.get("oof_path")]
                if len(ok) < 2:
                    issues.append(f"для бленда нужно ≥2 успешных эксперимента; из {ids} успешны {ok}")
                else:
                    impl["exp_ids"] = ok
        elif kind == "postprocess":
            if not self.ws.cfg.is_classification or self.ws.metric.needs != "label":
                issues.append("postprocess применим только к классификации с метрикой по меткам")
            base = self.store.get(impl.get("base_exp_id") or "")
            if base is None or base.get("status") != "success":
                impl["base_exp_id"] = (self.store.best() or {}).get("exp_id")
        if spec_signature(impl) in self.store.signatures():
            issues.append("такой эксперимент уже проводился — предложи другое изменение")
        if kind in ("registry", "code"):
            est = self.ws.rc.estimate(family_of(impl), work_units_of(impl))
            try:
                est = max(est, float(h.get("estimated_minutes") or 0) * 60 * 0.5)
            except (TypeError, ValueError):
                pass
            if not self.ws.rc.can_start(est):
                issues.append(f"не уложится во время: оценка {fmt_duration(est)}, на исследование осталось "
                              f"{fmt_duration(self.ws.rc.remaining_research())}; нужна более быстрая конфигурация")
        return issues

    # ------------------------------------------------------------------ LLM
    def _durations_text(self) -> str:
        fams: dict[str, list[float]] = {}
        for f, d, s, _ in self.ws.rc.durations:
            if s == "success":
                fams.setdefault(f, []).append(d)
        if not fams:
            return "длительности пока неизвестны"
        return "; ".join(f"{f} ≈ {fmt_duration(sum(v) / len(v))}" for f, v in fams.items())

    def _prompt(self) -> tuple[str, str]:
        ws = self.ws
        models = "\n".join(f"- {k}: {v}; дефолт {json.dumps(model_zoo.DEFAULTS.get(k, {}), ensure_ascii=False)}"
                           for k, v in self.models.items())
        system = prompts.HYPOTHESIS_SYSTEM.format(models=models, contract=prompts.CODE_CONTRACT)
        best = self.store.best()
        tried = [f"{h['exp_id']}: {h.get('kind')}:{h.get('model') or ''} {short(json.dumps((h.get('hypothesis') or {}).get('implementation', {}), ensure_ascii=False, default=str), 160)} → {h.get('status')} {fmt_num(h.get('local_cv_score'))}"
                 for h in self.store.history]
        extra = []
        cands = pick_blend_candidates(self.store)
        if len(cands) >= 2:
            extra.append(f"Успешные эксперименты для бленда: {cands}.")
        if ws.cfg.is_classification and ws.metric.needs == "label":
            extra.append("Метрика считается по меткам — постобработка весами классов (kind=postprocess) может помочь.")
        user = prompts.HYPOTHESIS_USER.format(
            task_summary=(f"{ws.cfg.task_description}\nТаргет '{ws.cfg.target}', тип {ws.cfg.task_type}, формат ответа "
                          f"{ws.cfg.prediction_format}." + (f"\nКлассы: {ws.meta.get('classes')[:30]}" if ws.cfg.is_classification else "")),
            eda_summary=self.eda_summary, validation=ws.plan.description,
            metric=ws.metric.name, direction=ws.metric.direction,
            best=(f"{best['exp_id']} = {fmt_num(best['local_cv_score'])} ({short(best['hypothesis_summary'], 150)})" if best else "нет"),
            top=history_table(sorted(self.store.successes(), key=lambda h: h["local_cv_score"],
                                     reverse=self.store.metric.greater_is_better)[:5], 5),
            history=history_table(self.store.history, 10), analysis=short(self.last_analysis, 2500) or "нет",
            resources=ws.rc.status_text() + "\nДлительность по видам: " + self._durations_text(),
            tried="\n".join(tried[-30:]) or "ничего", extra=("\n".join(extra) + "\n\n") if extra else "",
        )
        return system, user

    def _from_researcher(self) -> tuple[dict | None, str]:
        """Гипотеза от Researcher Agent. Возвращает (готовая гипотеза | None, направление для LLM-генератора)."""
        if self.researcher is None or not self.researcher.available():
            return None, ""
        extra = (f"Доступные модели (implementation): {list(self.models)}. "
                 f"Валидация фиксирована харнессом: {self.ws.plan.description}")
        raw = self.researcher.propose(self.store.history, self.last_reflection, extra)
        if not raw:
            return None, ""
        guidance = (f"Researcher Agent предлагает: {short(raw.get('statement'), 400)}\n"
                    f"Механизм: {short(raw.get('mechanism'), 300)}\nПлан: {short(raw.get('plan'), 500)}\n"
                    f"Вмешательства: {short(json.dumps(raw.get('interventions'), ensure_ascii=False, default=str), 600)}")
        h = normalize_hypothesis({k: v for k, v in raw.items() if k != "implementation"})
        h["summary"] = short(raw.get("statement") or h.get("summary"), 300)
        impl = h["implementation"]
        if impl.get("kind") == "registry" and not impl.get("model"):
            best = self.store.best()
            impl["model"] = (best or {}).get("model") or self.gbdt[0]
        issues = self.critic(h)
        if not issues:
            return h, guidance
        log(f"Гипотеза Researcher Agent не исполнима напрямую ({short('; '.join(issues), 300)}) — "
            f"передаю её LLM-генератору как направление")
        return None, guidance

    def _from_llm(self, guidance: str = "") -> dict | None:
        system, user = self._prompt()
        if guidance:
            user += ("\n\n## Направление от исследователя\n" + guidance +
                     "\nРеализуй эту идею (kind=code, если реестра недостаточно), если она не противоречит истории.")
        messages_feedback = ""
        for attempt in range(2):
            u = user + (f"\n\n{messages_feedback}" if messages_feedback else "")
            obj, raw = self.llm.ask_json(system, u, max_tokens=3000, temperature=0.5 if attempt == 0 else 0.3,
                                         stage="hypothesis", required_keys=("implementation",))
            if obj is None:
                self.llm_failures += 1
                return None
            impl = obj.get("implementation") or {}
            if isinstance(impl, dict) and impl.get("kind") == "code" and not impl.get("code"):
                impl["code"] = extract_code(raw)
            obj["implementation"] = impl
            h = normalize_hypothesis(obj)
            issues = self.critic(h)
            if not issues:
                return h
            log(f"Критик отклонил гипотезу LLM: {issues}")
            messages_feedback = prompts.CRITIC_FEEDBACK.format(issues="; ".join(issues))
        self.llm_failures += 1
        return None

    # ------------------------------------------------------------------ public
    def next(self) -> tuple[dict | None, str]:
        """Следующая гипотеза и её источник ("llm" | "ladder" | "auto")."""
        n_ok = len(self.store.successes())
        # первые шаги — сильные гипотезы из лестницы: они почти всегда полезны и дают базу для LLM
        if n_ok < 2 and self.ladder:
            h = self._next_fallback()
            if h is not None:
                return normalize_hypothesis(h), "ladder"
        guidance = ""
        h, guidance = self._from_researcher()
        if h is not None:
            return h, "researcher"
        if self.llm is not None and self.llm.available(12000) and self.llm_failures < 4:
            h = self._from_llm(guidance)
            if h is not None:
                return h, "llm"
        h = self._next_fallback()
        if h is None:
            return None, "none"
        impl = normalize_hypothesis(h)["implementation"]
        return normalize_hypothesis(h), ("auto" if impl["kind"] in ("blend", "postprocess") else "ladder")

    def repair(self, rec: dict) -> dict | None:
        """Одна попытка починить упавший код LLM по трейсбеку."""
        if self.llm is None or not self.llm.available(8000) or rec.get("status") != "error":
            return None
        hyp = rec.get("hypothesis") or {}
        code_path = self.ws.work_dir / "experiments" / rec["exp_id"] / "model_code.py"
        if not code_path.exists():
            return None
        code = code_path.read_text(encoding="utf-8")
        system = prompts.CODE_FIX_SYSTEM.format(contract=prompts.CODE_CONTRACT)
        user = prompts.CODE_FIX_USER.format(
            statement=short(hyp.get("statement") or rec.get("hypothesis_summary"), 600), code=code,
            error=short(rec.get("traceback") or rec.get("error_message"), 3000),
            columns=short(", ".join(self.ws.meta["cat_features"] + self.ws.meta["num_features"]), 1500),
            cat_features=short(", ".join(self.ws.meta["cat_features"]), 500))
        text = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             max_tokens=2500, temperature=0.2, stage="code_fix")
        new_code = extract_code(text)
        if not new_code:
            return None
        h = dict(hyp)
        impl = dict(h.get("implementation") or {})
        impl.update(kind="code", code=new_code)
        h["implementation"] = impl
        h["summary"] = short(f"{rec.get('hypothesis_summary')} (исправленный код)", 300)
        h = normalize_hypothesis(h)
        issues = self.critic(h)
        if issues:
            log(f"Исправленный код отклонён критиком: {issues}")
            return None
        return h
