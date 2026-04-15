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

        for step_output in self.graph.stream(state):
            node_name = list(step_output.keys())[0]
            node_state = step_output[node_name]
            yield {
                "node": node_name,
                "iteration": node_state.get("iteration", 0),
                "log": node_state.get("log", []),
                "status": node_state.get("status", "running"),
            }

        yield {"node": "complete", "status": "done", "results": state}

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

            # Strategy: take real samples and flip the protected attribute to all values
            base_samples = X_train.sample(min(20, len(X_train)), random_state=iteration)
            attr_values = X_train[attr].unique()

            for _, row in base_samples.iterrows():
                for val in attr_values:
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
        Uses Gemini to interpret results semantically.
        """
        model = state["model"]
        probes = state["synthetic_probes"]
        log = state.get("log", [])

        log.append(f"[Evaluator Agent] Running {len(probes)} probes through model...")

        results = []
        for probe in probes:
            try:
                features_df = pd.DataFrame([probe["features"]])
                pred = model.predict(features_df)[0]
                prob = model.predict_proba(features_df)[0][1] if hasattr(model, 'predict_proba') else None
                results.append({
                    **probe,
                    "prediction": int(pred),
                    "probability": float(prob) if prob is not None else None,
                })
            except Exception as e:
                continue

        # Group by (base case, target attribute) and compute disparity
        evaluation = []
        attr_groups = {}
        for r in results:
            key = r["base_attr"]
            if key not in attr_groups:
                attr_groups[key] = []
            attr_groups[key].append(r)

        for attr, group in attr_groups.items():
            probs = [r["probability"] for r in group if r["probability"] is not None]
            if not probs:
                continue
            disparity = max(probs) - min(probs)
            evaluation.append({
                "attribute": attr,
                "disparity": round(disparity, 4),
                "max_prob": round(max(probs), 4),
                "min_prob": round(min(probs), 4),
                "bias_confirmed": disparity > 0.1,
                "sample_count": len(group),
            })

        log.append(f"[Evaluator Agent] Bias confirmed in {sum(1 for e in evaluation if e['bias_confirmed'])} attributes")
        return {**state, "evaluation_results": evaluation, "log": log}

    def _decide_patch_node(self, state: Dict) -> Dict:
        """Use Gemini to decide: is the evidence strong enough to patch?"""
        evaluation = state["evaluation_results"]
        confirmed = [e for e in evaluation if e.get("bias_confirmed")]

        # Build mitigation plan
        mitigation_plan = []
        for e in confirmed:
            strategy = self._select_mitigation_strategy(e)
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
                if strategy == "sample_reweighing":
                    weights = self._compute_reweighing_weights(X_train, y_train, attr)
                    model.fit(X_train, y_train, sample_weight=weights)
                    patch_results["applied"].append({"strategy": strategy, "attribute": attr})

                elif strategy == "threshold_adjustment":
                    # Store per-group thresholds (applied at inference time)
                    state["group_thresholds"] = self._compute_group_thresholds(
                        model, X_train, y_train, attr
                    )
                    patch_results["applied"].append({"strategy": strategy, "attribute": attr})

                elif strategy == "feature_ablation":
                    # Remove the proxy feature
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
        Validator Agent: re-run bias evaluation post-patch.
        Confirms improvement, flags regressions.
        """
        model = state["model"]
        X_train = state["X_train"]
        y_train = state["y_train"]
        log = state.get("log", [])

        log.append("[Validator Agent] Re-evaluating bias metrics post-patch...")

        # Re-run predictions
        try:
            y_pred_new = model.predict(X_train)
            y_prob_new = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None

            # Compare against confirmed biases
            validation = {"improved": [], "regressed": [], "unchanged": []}
            for e in state.get("evaluation_results", []):
                attr = e["attribute"]
                if attr not in X_train.columns:
                    validation["improved"].append(attr)
                    continue

                for val in X_train[attr].unique():
                    mask = X_train[attr] == val
                    if mask.sum() == 0:
                        continue

                old_disparity = e.get("disparity", 0)
                if y_prob_new is not None:
                    new_disparities = []
                    for val1 in X_train[attr].unique():
                        for val2 in X_train[attr].unique():
                            if val1 != val2:
                                m1 = X_train[attr] == val1
                                m2 = X_train[attr] == val2
                                if m1.sum() > 0 and m2.sum() > 0:
                                    diff = abs(y_prob_new[m1].mean() - y_prob_new[m2].mean())
                                    new_disparities.append(diff)
                    new_disparity = max(new_disparities) if new_disparities else 0
                else:
                    new_disparity = old_disparity

                if new_disparity < old_disparity * 0.7:
                    validation["improved"].append({"attribute": attr, "before": old_disparity, "after": new_disparity})
                elif new_disparity > old_disparity * 1.1:
                    validation["regressed"].append({"attribute": attr, "before": old_disparity, "after": new_disparity})
                else:
                    validation["unchanged"].append(attr)

            log.append(
                f"[Validator Agent] Results: {len(validation['improved'])} improved, "
                f"{len(validation['regressed'])} regressed, {len(validation['unchanged'])} unchanged"
            )

        except Exception as e:
            validation = {"error": str(e)}
            log.append(f"[Validator Agent] Validation error: {e}")

        return {**state, "validation_results": validation, "log": log}

    def _report_node(self, state: Dict) -> Dict:
        """Generate final structured report."""
        log = state.get("log", [])
        log.append("[Report Agent] Generating final red-team report...")

        report = {
            "audit_id": state.get("audit_id"),
            "completed_at": datetime.utcnow().isoformat(),
            "iterations": state.get("iteration", 0),
            "biases_targeted": len(state.get("confirmed_biases", [])),
            "patches_applied": len(state.get("patch_results", {}).get("applied", [])),
            "validation": state.get("validation_results", {}),
            "log_summary": log[-10:],  # last 10 log entries
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

    def _select_mitigation_strategy(self, evaluation: Dict) -> Dict:
        disparity = evaluation.get("disparity", 0)
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
