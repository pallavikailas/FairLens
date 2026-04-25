"""
Model Bias Probe Service
========================
Probes the provided model against a neutral EMBEDDED reference dataset to reveal
hidden biases intrinsic to the model itself — completely independent of whatever
dataset the user has uploaded.

Pipeline
--------
1. Generate reference probe dataset (fixed 300-row dataset OR model-specific probe)
2. Run model predictions on reference dataset
3. Run Bias Cartography on (reference data + model predictions) → demographic disparity map
4. Run Counterfactual Constitution on (reference data + model) → implicit decision rules
5. Extract structured bias list for red-team targeting
"""

import io
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from app.services.reference_dataset import (
    generate_reference_dataset,
    generate_text_reference_dataset,
    generate_model_specific_probe,
    REFERENCE_PROTECTED_COLS,
    REFERENCE_TARGET_COL,
)
from app.services.cartography import cartography_service
from app.services.constitution import constitution_service

logger = logging.getLogger(__name__)

_LLM_TYPES = {"GenerativeLLM", "OpenAI", "Gemini", "HuggingFace"}


def _is_llm(model) -> bool:
    mt = (getattr(model, "get_model_type", lambda: "")() or "")
    return any(t in mt for t in _LLM_TYPES)


def _uses_text_reference_probe(model) -> bool:
    mt = (getattr(model, "get_model_type", lambda: "")() or "")
    return mt == "HuggingFace"


class ModelBiasProbe:

    async def probe(
        self,
        model: Any,
        model_type: str,
        audit_id: str,
        user_protected_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Probe *model* on the embedded reference dataset and return
        a structured dict of discovered hidden biases.
        """
        logger.info(f"[{audit_id}] Starting model bias probe with embedded reference dataset")

        # ── 1. Build probe dataset ────────────────────────────────────────────
        if _uses_text_reference_probe(model):
            ref_df, ref_csv = generate_text_reference_dataset()
            probe_protected = REFERENCE_PROTECTED_COLS
            probe_target = REFERENCE_TARGET_COL
        elif _is_llm(model):
            # LLM models handle any text; use the fixed standard reference dataset
            ref_df, ref_csv = generate_reference_dataset()
            probe_protected = REFERENCE_PROTECTED_COLS
            probe_target    = REFERENCE_TARGET_COL
        else:
            # Structured (sklearn / API) models: generate a dataset matching model features
            feature_names = self._get_feature_names(model)
            if feature_names:
                ref_df, ref_csv, probe_protected, probe_target = generate_model_specific_probe(
                    feature_names, protected_cols=user_protected_cols
                )
            else:
                # Fallback to standard reference dataset
                ref_df, ref_csv = generate_reference_dataset()
                probe_protected = REFERENCE_PROTECTED_COLS
                probe_target    = REFERENCE_TARGET_COL

        logger.info(
            f"[{audit_id}] Probe dataset: {len(ref_df)} rows, "
            f"protected={probe_protected}, target='{probe_target}'"
        )

        # ── 2. Get model predictions on reference dataset ────────────────────
        feature_cols = [c for c in ref_df.columns if c != probe_target]
        X_ref = ref_df[feature_cols]

        try:
            raw_preds = model.predict(X_ref)
            model_predictions = [int(p) for p in raw_preds]
            logger.info(f"[{audit_id}] Model generated {len(model_predictions)} predictions on reference dataset")
        except Exception as e:
            logger.error(f"[{audit_id}] Model prediction on reference dataset failed: {e}")
            raise ValueError(f"Model could not be probed on reference dataset: {e}")

        diagnostics = self._prediction_diagnostics(
            ref_df=ref_df,
            predictions=model_predictions,
            target_col=probe_target,
        )

        # ── 3. Bias Cartography on reference data + model predictions ────────
        carto_results = await cartography_service.run_cartography(
            dataset_csv=ref_csv,
            protected_cols=probe_protected,
            target_col=probe_target,
            model_predictions=model_predictions,
            audit_id=audit_id,
        )
        if diagnostics["collapsed_output"] or diagnostics["near_constant_output"]:
            carto_results["fair_score"] = {
                "score": 0,
                "label": "Invalid",
                "color": "red",
                "reason": diagnostics["reason"],
            }

        # ── 4. Counterfactual Constitution on reference data + model ─────────
        try:
            y_pred_arr = np.array(model_predictions)
            constitution_results = await constitution_service.generate_constitution(
                model=model,
                X=X_ref,
                y_pred=y_pred_arr,
                protected_cols=probe_protected,
                feature_names=feature_cols,
                cartography_results=carto_results,
                audit_id=audit_id,
            )
        except Exception as e:
            logger.warning(f"[{audit_id}] Constitution on reference dataset failed: {e}")
            constitution_results = {"error": str(e), "patterns": [], "sections": []}

        # ── 5. Extract structured bias list ──────────────────────────────────
        model_biases = self._extract_model_biases(carto_results, constitution_results)
        if diagnostics["collapsed_output"] or diagnostics["near_constant_output"]:
            model_biases.insert(0, {
                "attribute": "model_output_distribution",
                "value": diagnostics["reason"],
                "type": "model_failure",
                "severity": "critical",
                "magnitude": 1.0,
                "source": "model_probe_diagnostics",
                "positive_rate": diagnostics["positive_rate"],
                "accuracy_vs_reference": diagnostics["accuracy_vs_reference"],
            })

        return {
            "audit_id":              audit_id,
            "analysis_type":         "model_probe",
            "reference_dataset_size": len(ref_df),
            "reference_protected_cols": probe_protected,
            "reference_target_col":  probe_target,
            "cartography":           carto_results,
            "constitution":          constitution_results,
            "prediction_diagnostics": diagnostics,
            "model_biases":          model_biases,
            "summary": {
                "fair_score":             carto_results.get("fair_score", {}),
                "bias_count":             len(model_biases),
                "most_biased_attribute":  model_biases[0]["attribute"] if model_biases else None,
                "analysis_source":        "embedded_reference_dataset",
                "model_type":             model_type,
                "prediction_diagnostics": diagnostics,
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_feature_names(model) -> Optional[List[str]]:
        """Try to extract feature names from sklearn-style model."""
        raw = getattr(model, "_model", model)
        for attr in ("feature_names_in_", "feature_names"):
            val = getattr(raw, attr, None)
            if val is not None:
                return list(val)
        try:
            val = raw.feature_name_()
            if val:
                return list(val)
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_model_biases(carto: Dict, constitution: Dict) -> List[Dict]:
        """Collect structured bias objects from cartography hotspots + constitution patterns."""
        biases: Dict[str, Dict] = {}

        for hotspot in carto.get("hotspots", []):
            attr = str(hotspot.get("dominant_slice", "")).split("=")[0].strip()
            if not attr:
                continue
            entry = {
                "attribute": attr,
                "value":     hotspot.get("dominant_slice", ""),
                "type":      "statistical_disparity",
                "severity":  hotspot.get("severity", "medium"),
                "magnitude": float(hotspot.get("mean_bias_magnitude", 0)),
                "source":    "model_probe_cartography",
                "spd":       float(hotspot.get("statistical_parity_diff", 0)),
            }
            if attr not in biases or entry["magnitude"] > biases[attr]["magnitude"]:
                biases[attr] = entry

        for pattern in constitution.get("patterns", []):
            attr = pattern.get("attribute", "")
            if not attr or pattern.get("flip_rate", 0) <= 0.05:
                continue
            entry = {
                "attribute": attr,
                "value":     pattern.get("bias_direction", ""),
                "type":      "counterfactual_flip",
                "severity":  pattern.get("severity", "medium"),
                "magnitude": float(pattern.get("flip_rate", 0)),
                "source":    "model_probe_constitution",
                "flip_rate": float(pattern.get("flip_rate", 0)),
            }
            if attr not in biases or entry["magnitude"] > biases[attr]["magnitude"]:
                biases[attr] = entry

        return sorted(biases.values(), key=lambda x: x["magnitude"], reverse=True)

    @staticmethod
    def _prediction_diagnostics(ref_df: pd.DataFrame, predictions: List[int], target_col: str) -> Dict[str, Any]:
        pred = np.array(predictions, dtype=int)
        unique_values = sorted(np.unique(pred).tolist())
        positive_rate = float(pred.mean()) if len(pred) else 0.0
        collapsed_output = len(unique_values) <= 1
        near_constant_output = positive_rate <= 0.01 or positive_rate >= 0.99

        accuracy_vs_reference = None
        if target_col in ref_df.columns:
            y_true = pd.to_numeric(ref_df[target_col], errors="coerce").fillna(0).astype(int).values
            if len(y_true) == len(pred):
                accuracy_vs_reference = round(float((y_true == pred).mean()), 4)

        reason = ""
        if collapsed_output:
            label = unique_values[0] if unique_values else 0
            reason = f"Model predicted a single class ({label}) for every reference sample."
        elif near_constant_output:
            reason = f"Model output was near-constant across the reference probe (positive rate {positive_rate:.3f})."

        return {
            "unique_prediction_count": len(unique_values),
            "unique_prediction_values": unique_values,
            "positive_rate": round(positive_rate, 4),
            "collapsed_output": collapsed_output,
            "near_constant_output": near_constant_output,
            "accuracy_vs_reference": accuracy_vs_reference,
            "reason": reason,
        }


model_probe_service = ModelBiasProbe()
