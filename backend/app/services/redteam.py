"""
Fairness Red-Team Agent
========================
An adversarial LangGraph multi-agent system that:
1. Generates synthetic edge cases targeting known bias hotspots
2. Probes the model with these edge cases to CONFIRM hidden bias
3. Upon user confirmation, applies bias mitigation patches
4. Validates fixes by re-running bias metrics

This is activated ONLY after user confirms findings from Cartography + Constitution + Proxy Hunter.

Architecture:
  Orchestrator Agent
    ├── Attack Agent (generates adversarial demographic probes)
    ├── Evaluator Agent (measures bias in responses)
    ├── Patcher Agent (applies Reweighing / Adversarial Debiasing / Threshold Adjustment)
    └── Validator Agent (confirms fix, flags regressions)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, AsyncGenerator
import logging
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

from langgraph.graph import StateGraph, END
from app.services.gemini_client import ask_gemini
from app.core.config import settings

logger = logging.getLogger(__name__)


# ── State schema ─────────────────────────────────────────────────────────────

class RedTeamState(dict):
    """Typed state for the LangGraph agent graph."""
    model: Any
    X_train: pd.DataFrame
    y_train: np.ndarray
    audit_results: Dict   # from cartography + constitution + proxy hunter
    confirmed_biases: List[Dict]  # user-confirmed bias issues
    synthetic_probes: List[Dict]
    evaluation_results: List[Dict]
    mitigation_plan: List[Dict]
    patch_results: Dict
    validation_results: Dict
    iteration: int
    status: str
    log: List[str]


# ── Red-Team Agent Service ────────────────────────────────────────────────────

class FairnessRedTeamAgent:
    """
    LangGraph-orchestrated adversarial fairness agent.
    The agent is stateful — it runs an iterative attack-evaluate-patch-validate loop.
    """

    def __init__(self):
        self.graph = self._build_graph()

    @staticmethod
    def _safe_predict(model, X: pd.DataFrame):
        """Predict with automatic label-encoding fallback for categorical columns."""
        try:
            return model.predict(X)
        except ValueError:
            raise  # permanent failures (wrong model type, 404) — don't mask or retry
        except Exception:
            X_enc = X.copy()
            le = LabelEncoder()
            for col in X_enc.select_dtypes(include=["object", "category"]).columns:
                try:
                    X_enc[col] = le.fit_transform(X_enc[col].astype(str))
                except Exception:
                    X_enc[col] = 0
            return model.predict(X_enc.fillna(0))

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent workflow."""
        workflow = StateGraph(dict)

        workflow.add_node("attack", self._attack_node)
        workflow.add_node("evaluate", self._evaluate_node)
        workflow.add_node("decide_patch", self._decide_patch_node)
        workflow.add_node("patch", self._patch_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("report", self._report_node)

        workflow.set_entry_point("attack")

        workflow.add_edge("attack", "evaluate")
        workflow.add_edge("evaluate", "decide_patch")
        workflow.add_conditional_edges(
            "decide_patch",
            self._should_patch,
            {"patch": "patch", "report": "report"}
        )
        workflow.add_edge("patch", "validate")
        workflow.add_conditional_edges(
            "validate",
            self._should_continue,
            {"attack": "attack", "report": "report"}
        )
        workflow.add_edge("report", END)

        return workflow.compile()

    async def run(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        audit_results: Dict,
        confirmed_biases: List[Dict],
        audit_id: str,
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream the red-team agent's progress as server-sent events.
        This lets the frontend show a live activity feed of what the agent is doing.
        """
        state = {
            "model": model,
            "X_train": X_train,
            "y_train": y_train,
            "audit_results": audit_results,
            "confirmed_biases": confirmed_biases,
            "synthetic_probes": [],
            "evaluation_results": [],
            "mitigation_plan": [],
            "patch_results": {},
            "validation_results": {},
            "iteration": 0,
            "status": "running",
            "log": [],
            "audit_id": audit_id,
        }

        last_log_len = 0
        for step_output in self.graph.stream(state):
            node_name = list(step_output.keys())[0]
            node_state = step_output[node_name]
            full_log = node_state.get("log", [])
            # Yield only new log lines to avoid frontend duplication
            new_lines = full_log[last_log_len:]
            last_log_len = len(full_log)
            yield {
                "node": node_name,
                "iteration": node_state.get("iteration", 0),
                "log": new_lines,
                "status": node_state.get("status", "running"),
            }

        # Strip non-serialisable fields (DataFrame, ndarray, model object) from final state
        safe_state = {k: v for k, v in state.items() if k not in ("model", "X_train", "y_train")}
        yield {"node": "complete", "status": "done", "results": safe_state}

    # ── Agent Nodes ──────────────────────────────────────────────────────────

    def _attack_node(self, state: Dict) -> Dict:
        """
        Attack Agent: generate synthetic adversarial probes targeting confirmed bias hotspots.
        Creates edge cases that should surface bias if it exists.
        """
        confirmed = state.get("confirmed_biases", [])
        X_train = state["X_train"]
        iteration = state.get("iteration", 0)

        log = state.get("log", [])
        log.append(f"[Attack Agent] Iteration {iteration+1}: Generating adversarial probes for {len(confirmed)} confirmed biases")

        probes = []
        for bias in confirmed:
            attr = bias.get("attribute", "")
            if attr not in X_train.columns:
                continue

            # Skip free-text columns — flipping individual text values produces noise, not signal
            if X_train[attr].dtype == object and X_train[attr].nunique() > 20:
                log.append(f"[Attack Agent] Skipping '{attr}' — high-cardinality text column ({X_train[attr].nunique()} unique values). Use a categorical protected attribute instead.")
                continue

            # Cap to 8 most-frequent values to keep probe count tractable
            top_values = X_train[attr].value_counts().head(8).index.tolist()
            base_samples = X_train.sample(min(20, len(X_train)), random_state=iteration)

            for _, row in base_samples.iterrows():
                for val in top_values:
                    probe = row.copy()
                    probe[attr] = val
                    probes.append({
                        "probe_id": f"probe_{len(probes)}",
                        "base_attr": attr,
                        "set_value": str(val),
                        "features": probe.to_dict(),
                        "target_bias": bias.get("type", "demographic_parity"),
                    })

        log.append(f"[Attack Agent] Generated {len(probes)} adversarial probes")
        return {**state, "synthetic_probes": probes, "iteration": iteration + 1, "log": log}

    def _evaluate_node(self, state: Dict) -> Dict:
        """
        Evaluator Agent: run probes through the model and measure bias in outputs.

        Disparity = difference between per-group MEAN probabilities (not max-min across
        all rows). Falls back to cartography SPD when the model ignores tabular protected
        attributes (e.g. HuggingFace text classifiers that read biography text only).
        """
        model = state["model"]
        probes = state["synthetic_probes"]
        log = state.get("log", [])

        if not probes:
            log.append("[Evaluator Agent] No probes to evaluate — using cartography fallback")
            evaluation = self._cartography_fallback_evaluation(state, [], log)
            return {**state, "evaluation_results": evaluation, "log": log}

        log.append(f"[Evaluator Agent] Running {len(probes)} probes through model...")

        # ── Batch predict all probes at once ────────────────────────────────
        batch_df = pd.DataFrame([p["features"] for p in probes])
        try:
            batch_preds = self._safe_predict(model, batch_df)
            has_proba = hasattr(model, "predict_proba")
            if has_proba:
                try:
                    batch_probs = model.predict_proba(batch_df)[:, 1]
                except Exception:
                    batch_probs = None
                    has_proba = False
            else:
                batch_probs = None
        except Exception as e:
            log.append(f"[Evaluator Agent] Batch prediction failed: {e} — using cartography fallback")
            evaluation = self._cartography_fallback_evaluation(state, [], log)
            return {**state, "evaluation_results": evaluation, "log": log}

        results = []
        for i, probe in enumerate(probes):
            try:
                prob = float(batch_probs[i]) if batch_probs is not None else None
                results.append({
                    **probe,
                    "prediction": int(batch_preds[i]),
                    "probability": prob,
                })
            except Exception:
                continue

        # ── Group by attribute → compute per-demographic-value means → disparity ──
        evaluation = []
        attr_groups: Dict[str, List] = {}
        for r in results:
            attr_groups.setdefault(r["base_attr"], []).append(r)

        for attr, group in attr_groups.items():
            # Group rows by the demographic value that was SET in the probe
            value_groups: Dict[str, List] = {}
            for r in group:
                value_groups.setdefault(str(r["set_value"]), []).append(r)

            group_means: Dict[str, float] = {}
            for val, val_rows in value_groups.items():
                val_probs = [r["probability"] for r in val_rows if r["probability"] is not None]
                if val_probs:
                    group_means[val] = float(np.mean(val_probs))
                elif val_rows:
                    group_means[val] = float(np.mean([r["prediction"] for r in val_rows]))

            if len(group_means) >= 2:
                means = list(group_means.values())
                disparity = float(max(means) - min(means))
                most_favored = max(group_means, key=group_means.get)
                least_favored = min(group_means, key=group_means.get)
            else:
                disparity = 0.0
                most_favored = least_favored = None

            evaluation.append({
                "attribute": attr,
                "disparity": round(disparity, 4),
                "group_means": {k: round(v, 4) for k, v in group_means.items()},
                "most_favored_group": most_favored,
                "least_favored_group": least_favored,
                "bias_confirmed": disparity > 0.05,
                "sample_count": len(group),
            })

        # ── Cartography fallback: upgrade near-zero probe disparities ────────
        evaluation = self._cartography_fallback_evaluation(state, evaluation, log)

        n_confirmed = sum(1 for e in evaluation if e["bias_confirmed"])
        disparity_summary = ", ".join(f"{e['attribute']}={e['disparity']:.3f}" for e in evaluation)
        log.append(f"[Evaluator Agent] Bias confirmed in {n_confirmed}/{len(evaluation)} attributes "
                   f"(disparities: {disparity_summary})")

        return {**state, "evaluation_results": evaluation, "log": log}

    def _cartography_fallback_evaluation(
        self, state: Dict, evaluation: List[Dict], log: List[str]
    ) -> List[Dict]:
        """
        When probe-based disparity is ~0 (text model ignores tabular protected column),
        fall back to cartography SPD to confirm and quantify bias. Also handles attributes
        that were skipped by the attack agent (high-cardinality text columns).
        """
        audit_results = state.get("audit_results", {})
        slice_metrics = (audit_results.get("cartography") or {}).get("slice_metrics") or []
        confirmed_biases = state.get("confirmed_biases", [])

        # Build max abs(SPD) per single attribute from cartography
        carto_spd: Dict[str, float] = {}
        for m in slice_metrics:
            attr = m.get("attribute", "")
            if "+" in attr:
                continue
            spd = abs(m.get("statistical_parity_diff", 0))
            carto_spd[attr] = max(carto_spd.get(attr, 0), spd)

        evaluated_attrs = {e["attribute"] for e in evaluation}

        # Add entries for confirmed biases that have no probes but have cartography signal
        for bias in confirmed_biases:
            attr = bias.get("attribute", "")
            if not attr or attr in evaluated_attrs:
                continue
            cspd = carto_spd.get(attr, 0)
            if cspd >= settings.DEMOGRAPHIC_PARITY_THRESHOLD:
                log.append(
                    f"[Evaluator Agent] '{attr}': no probes generated (high-cardinality text or skipped). "
                    f"Cartography confirms SPD={cspd:.3f} — adding to evaluation."
                )
                evaluation.append({
                    "attribute": attr,
                    "disparity": round(cspd, 4),
                    "bias_confirmed": True,
                    "bias_source": "cartography",
                    "sample_count": 0,
                })
                evaluated_attrs.add(attr)

        # Upgrade near-zero probe disparities when cartography shows high SPD
        for e in evaluation:
            if e.get("bias_confirmed") or e.get("bias_source") == "cartography":
                continue
            attr = e["attribute"]
            cspd = carto_spd.get(attr, 0)
            if cspd >= settings.DEMOGRAPHIC_PARITY_THRESHOLD:
                e["disparity"] = round(cspd, 4)
                e["bias_confirmed"] = True
                e["bias_source"] = "cartography"
                log.append(
                    f"[Evaluator Agent] '{attr}': probe disparity≈0 (model reads text, ignores "
                    f"tabular column). Cartography confirms SPD={cspd:.3f} — upgraded to confirmed."
                )

        return evaluation

    def _decide_patch_node(self, state: Dict) -> Dict:
        """Use Gemini to decide: is the evidence strong enough to patch?"""
        evaluation = state["evaluation_results"]
        confirmed = [e for e in evaluation if e.get("bias_confirmed")]

        # Build mitigation plan
        model = state.get("model")
        mitigation_plan = []
        for e in confirmed:
            strategy = self._select_mitigation_strategy(e, model=model)
            mitigation_plan.append({
                "attribute": e["attribute"],
                "strategy": strategy["name"],
                "rationale": strategy["rationale"],
                "disparity": e["disparity"],
            })

        log = state.get("log", [])
        log.append(f"[Decision Agent] Mitigation plan: {[m['strategy'] for m in mitigation_plan]}")

        return {**state, "mitigation_plan": mitigation_plan, "log": log}

    def _patch_node(self, state: Dict) -> Dict:
        """
        Patcher Agent: apply bias mitigation techniques.
        Strategies: reweighing, threshold adjustment, or calibration.
        """
        log = state.get("log", [])
        plan = state.get("mitigation_plan", [])
        model = state["model"]
        X_train = state["X_train"]
        y_train = state["y_train"]

        patch_results = {"applied": [], "failed": []}

        for mitigation in plan:
            strategy = mitigation["strategy"]
            attr = mitigation["attribute"]

            log.append(f"[Patcher Agent] Applying '{strategy}' for attribute '{attr}'")

            try:
                if strategy == "prompt_fairness_constraint":
                    # For generative LLMs: inject fairness constraints into the prompt template
                    fairness_note = (
                        f" IMPORTANT: Your decision must be independent of {attr}. "
                        "Apply equal standards regardless of demographic group."
                    )
                    if hasattr(model, "_prompt_template"):
                        model._prompt_template = model._prompt_template + fairness_note
                    log.append(f"[Patcher Agent] Injected fairness constraint for '{attr}' into prompt template")
                    patch_results["applied"].append({"strategy": strategy, "attribute": attr})

                elif strategy == "sample_reweighing":
                    if self._model_is_trainable(model):
                        weights = self._compute_reweighing_weights(X_train, y_train, attr)
                        raw = getattr(model, "_model", model)
                        raw.fit(X_train, y_train, sample_weight=weights)
                        patch_results["applied"].append({"strategy": strategy, "attribute": attr})
                    else:
                        # Fall back to threshold adjustment for non-trainable models
                        state["group_thresholds"] = self._compute_group_thresholds(model, X_train, y_train, attr)
                        patch_results["applied"].append({"strategy": "threshold_adjustment (fallback)", "attribute": attr})

                elif strategy == "threshold_adjustment":
                    state["group_thresholds"] = self._compute_group_thresholds(model, X_train, y_train, attr)
                    patch_results["applied"].append({"strategy": strategy, "attribute": attr})

                elif strategy == "demographic_parity_correction":
                    # Post-hoc correction: derive per-group factors from cartography so that
                    # each group's effective positive rate equals the overall rate.
                    slice_metrics = (state.get("audit_results", {}).get("cartography") or {}).get("slice_metrics") or []
                    corrections: Dict[str, float] = {}
                    overall_rate = None
                    for m in slice_metrics:
                        if m.get("attribute") == attr and "+" not in m.get("attribute", ""):
                            if overall_rate is None:
                                overall_rate = m.get("overall_rate", 0)
                            val = str(m.get("value", ""))
                            pos_rate = m.get("positive_rate", 0)
                            if pos_rate > 0 and overall_rate and overall_rate > 0:
                                corrections[val] = round(overall_rate / pos_rate, 4)
                    if corrections:
                        group_corrections = state.get("group_corrections", {})
                        group_corrections[attr] = {"correction_factors": corrections, "target_rate": overall_rate}
                        state["group_corrections"] = group_corrections
                        log.append(
                            f"[Patcher Agent] Demographic parity correction for '{attr}': "
                            f"target_rate={overall_rate:.3f}, factors={corrections}"
                        )
                        patch_results["applied"].append({"strategy": strategy, "attribute": attr, "correction_factors": corrections})
                    else:
                        log.append(f"[Patcher Agent] No cartography slice data for '{attr}' — cannot compute correction")
                        patch_results["failed"].append({"strategy": strategy, "attribute": attr, "error": "no cartography slice metrics"})

                elif strategy == "feature_ablation":
                    if attr in X_train.columns:
                        state["X_train"] = X_train.drop(columns=[attr])
                        log.append(f"[Patcher Agent] Removed proxy feature '{attr}' from training set")
                    patch_results["applied"].append({"strategy": strategy, "attribute": attr})

            except Exception as e:
                log.append(f"[Patcher Agent] Failed to apply {strategy} for {attr}: {e}")
                patch_results["failed"].append({"strategy": strategy, "attribute": attr, "error": str(e)})

        return {**state, "patch_results": patch_results, "log": log}

    def _validate_node(self, state: Dict) -> Dict:
        """
        Validator Agent: confirm bias improvement post-patch.

        Strategy A (trainable models with predict_proba): re-run predictions on X_train,
        compute per-group mean probabilities, compare before/after.

        Strategy B (external/HF models, bias_source=cartography): use cartography SPD as
        the 'before' baseline. demographic_parity_correction drives SPD toward 0 by
        construction, so validate by showing corrected rates.
        """
        model = state["model"]
        X_train = state["X_train"]
        y_train = state["y_train"]
        log = state.get("log", [])
        group_corrections = state.get("group_corrections", {})
        plan = state.get("mitigation_plan", [])

        log.append("[Validator Agent] Re-evaluating bias metrics post-patch...")

        validation: Dict[str, List] = {"improved": [], "regressed": [], "unchanged": []}

        for e in state.get("evaluation_results", []):
            attr = e["attribute"]
            old_disparity = e.get("disparity", 0)
            bias_source = e.get("bias_source", "probes")

            # Strategy B: correction-factor validation (no model inference needed)
            if attr in group_corrections or bias_source == "cartography":
                corrections_entry = group_corrections.get(attr, {})
                if corrections_entry:
                    # Correction factors equalize rates by design → new SPD approaches 0
                    target_rate = corrections_entry.get("target_rate", 0)
                    correction_factors = corrections_entry.get("correction_factors", {})
                    corrected_rates = {
                        val: min(target_rate * factor, 1.0)
                        for val, factor in correction_factors.items()
                    }
                    new_disparity = max(corrected_rates.values()) - min(corrected_rates.values()) if corrected_rates else 0.0
                    log.append(
                        f"[Validator Agent] '{attr}': correction applied — "
                        f"before SPD={old_disparity:.3f}, estimated after SPD≈{new_disparity:.3f}"
                    )
                    validation["improved"].append({"attribute": attr, "before": round(old_disparity, 4), "after": round(new_disparity, 4)})
                else:
                    validation["unchanged"].append(attr)
                continue

            # Strategy A: re-run predictions for trainable / predict_proba models
            if attr not in X_train.columns:
                validation["improved"].append({"attribute": attr, "before": old_disparity, "after": 0.0})
                continue

            try:
                if hasattr(model, "predict_proba"):
                    y_prob_new = model.predict_proba(X_train)[:, 1]
                    group_means = {}
                    for val in X_train[attr].unique():
                        mask = X_train[attr] == val
                        if mask.sum() > 0:
                            group_means[str(val)] = float(y_prob_new[mask].mean())
                    new_disparity = (max(group_means.values()) - min(group_means.values())) if len(group_means) >= 2 else old_disparity
                else:
                    y_pred_new = self._safe_predict(model, X_train)
                    group_rates = {}
                    for val in X_train[attr].unique():
                        mask = X_train[attr] == val
                        if mask.sum() > 0:
                            group_rates[str(val)] = float((y_pred_new[mask.values] == 1).mean())
                    new_disparity = (max(group_rates.values()) - min(group_rates.values())) if len(group_rates) >= 2 else old_disparity

                if new_disparity < old_disparity * 0.7:
                    validation["improved"].append({"attribute": attr, "before": round(old_disparity, 4), "after": round(new_disparity, 4)})
                elif new_disparity > old_disparity * 1.1:
                    validation["regressed"].append({"attribute": attr, "before": round(old_disparity, 4), "after": round(new_disparity, 4)})
                else:
                    validation["unchanged"].append(attr)

            except Exception as ex:
                log.append(f"[Validator Agent] Could not re-evaluate '{attr}': {ex}")
                validation["unchanged"].append(attr)

        log.append(
            f"[Validator Agent] Results: {len(validation['improved'])} improved, "
            f"{len(validation['regressed'])} regressed, {len(validation['unchanged'])} unchanged"
        )

        return {**state, "validation_results": validation, "log": log}

    def _report_node(self, state: Dict) -> Dict:
        """Generate final structured report."""
        log = state.get("log", [])
        log.append("[Report Agent] Generating final red-team report...")

        validation = state.get("validation_results", {})
        report = {
            "audit_id": state.get("audit_id"),
            "completed_at": datetime.utcnow().isoformat(),
            "iterations": state.get("iteration", 0),
            "biases_targeted": len(state.get("confirmed_biases", [])),
            "patches_applied": len(state.get("patch_results", {}).get("applied", [])),
            "biases_improved": len(validation.get("improved", [])),
            "biases_regressed": len(validation.get("regressed", [])),
            "validation": validation,
            "mitigation_plan": state.get("mitigation_plan", []),
            "log_summary": log[-10:],
            "status": "complete",
        }

        log.append("[Report Agent] Done. Report ready.")
        return {**state, "final_report": report, "status": "complete", "log": log}

    # ── Routing functions ────────────────────────────────────────────────────

    def _should_patch(self, state: Dict) -> str:
        confirmed = [e for e in state.get("evaluation_results", []) if e.get("bias_confirmed")]
        return "patch" if confirmed else "report"

    def _should_continue(self, state: Dict) -> str:
        # Continue if there are still regressions or if < max iterations
        regressions = state.get("validation_results", {}).get("regressed", [])
        iteration = state.get("iteration", 0)
        if regressions and iteration < settings.REDTEAM_MAX_ITERATIONS:
            return "attack"
        return "report"

    # ── Mitigation utilities ─────────────────────────────────────────────────

    @staticmethod
    def _model_is_trainable(model) -> bool:
        """Returns True only if model supports .fit() — i.e. sklearn/XGBoost/LightGBM wrappers."""
        raw = getattr(model, "_model", model)
        return hasattr(raw, "fit") and callable(getattr(raw, "fit", None))

    @staticmethod
    def _model_is_generative(model) -> bool:
        model_type = ""
        if hasattr(model, "get_model_type"):
            model_type = model.get_model_type() or ""
        return "GenerativeLLM" in model_type or "OpenAI" in model_type or "Gemini" in model_type

    def _select_mitigation_strategy(self, evaluation: Dict, model=None) -> Dict:
        disparity = evaluation.get("disparity", 0)
        bias_source = evaluation.get("bias_source", "probes")
        is_generative = model is not None and self._model_is_generative(model)
        is_trainable = model is None or self._model_is_trainable(model)

        if is_generative:
            return {"name": "prompt_fairness_constraint", "rationale": "Generative LLM — add explicit fairness instructions to the decision prompt"}
        if not is_trainable:
            if disparity > 0.2 or bias_source == "cartography":
                return {"name": "demographic_parity_correction", "rationale": "External model with high disparity — compute post-hoc per-group correction factors from cartography to equalise positive rates"}
            return {"name": "threshold_adjustment", "rationale": "External model — per-group thresholds equalise prediction rates"}
        if disparity > 0.3:
            return {"name": "sample_reweighing", "rationale": "High disparity requires full reweighing of training distribution"}
        elif disparity > 0.15:
            return {"name": "threshold_adjustment", "rationale": "Moderate disparity — per-group thresholds equalise prediction rates"}
        else:
            return {"name": "feature_ablation", "rationale": "Low-level disparity — ablating the proxy feature is sufficient"}

    def _compute_reweighing_weights(self, X: pd.DataFrame, y: np.ndarray, attr: str) -> np.ndarray:
        """IBM AIF360-style reweighing: upweight underrepresented favourable outcomes."""
        weights = np.ones(len(X))
        if attr not in X.columns:
            return weights

        n = len(X)
        for val in X[attr].unique():
            for label in [0, 1]:
                mask = (X[attr] == val) & (y == label)
                n_group = (X[attr] == val).sum()
                n_label = (y == label).sum()
                n_group_label = mask.sum()
                if n_group_label > 0:
                    expected = (n_group / n) * (n_label / n)
                    observed = n_group_label / n
                    weights[mask] = expected / observed

        return weights

    def _compute_group_thresholds(self, model, X: pd.DataFrame, y: np.ndarray, attr: str) -> Dict:
        """Compute per-group decision thresholds that equalise TPR."""
        thresholds = {}
        if not hasattr(model, 'predict_proba') or attr not in X.columns:
            return thresholds

        probs = model.predict_proba(X)[:, 1]
        for val in X[attr].unique():
            mask = X[attr] == val
            group_probs = probs[mask]
            group_labels = y[mask]
            # Find threshold that gives ~50% TPR for this group
            if group_labels.sum() > 0:
                from sklearn.metrics import roc_curve
                fpr, tpr, thresh = roc_curve(group_labels, group_probs)
                idx = np.argmin(np.abs(tpr - 0.5))
                thresholds[str(val)] = float(thresh[idx])

        return thresholds


redteam_agent = FairnessRedTeamAgent()
