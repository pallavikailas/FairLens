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
        if _is_llm(model):
            # LLM models handle any text; use the fixed standard reference dataset
            ref_df, ref_csv = generate_reference_dataset()
            probe_protected = REFERENCE_PROTECTED_COLS
            probe_target    = REFERENCE_TARGET_COL
        else:
            # Structured (sklearn / API) models: generate a dataset matching model features
            feature_names = self._get_feature_names(model)
            if feature_names:
                ref_df, ref_csv, probe_protected, probe_target, model_predict_cols = (
                    generate_model_specific_probe(feature_names, protected_cols=user_protected_cols)
                )
            else:
                # Fallback to standard reference dataset
                ref_df, ref_csv = generate_reference_dataset()
                probe_protected    = REFERENCE_PROTECTED_COLS
                probe_target       = REFERENCE_TARGET_COL
                model_predict_cols = None  # will use all non-target cols

        logger.info(
            f"[{audit_id}] Probe dataset: {len(ref_df)} rows, "
            f"protected={probe_protected}, target='{probe_target}'"
        )

        # ── 2. Get model predictions on reference dataset ────────────────────
        # model_predict_cols may be a subset when demographics were injected
        if _is_llm(model) or model_predict_cols is None:
            feature_cols = [c for c in ref_df.columns if c != probe_target]
        else:
            feature_cols = model_predict_cols
        X_ref = ref_df[feature_cols]

        try:
            raw_preds = model.predict(X_ref)
            model_predictions = [int(p) for p in raw_preds]
            logger.info(f"[{audit_id}] Model generated {len(model_predictions)} predictions on reference dataset")
        except Exception as e:
            logger.error(f"[{audit_id}] Model prediction on reference dataset failed: {e}")
            raise ValueError(f"Model could not be probed on reference dataset: {e}")

        # ── Degenerate prediction guard ───────────────────────────────────────
        # If the model predicts >97% the same class, the synthetic probe data is
        # out-of-distribution and any bias numbers would be statistically meaningless.
        n_pos = sum(model_predictions)
        n_tot = len(model_predictions)
        pos_rate = n_pos / n_tot if n_tot else 0.5
        if pos_rate <= 0.03 or pos_rate >= 0.97:
            majority_class = 1 if pos_rate > 0.5 else 0
            logger.warning(
                f"[{audit_id}] Degenerate probe: {pos_rate:.1%} of predictions = class {majority_class}. "
                "Synthetic reference data is likely out-of-distribution."
            )
            return {
                "audit_id":               audit_id,
                "analysis_type":          "model_probe",
                "degenerate":             True,
                "degenerate_message": (
                    f"Model predicted class {majority_class} for {pos_rate:.0%} of the synthetic reference "
                    f"dataset ({n_tot} rows). This means the probe data does not match the model's training "
                    "distribution — any bias numbers computed from it would be statistically meaningless. "
                    "To properly probe this model, upload a representative dataset so Phase 3 (Cross-Analysis) "
                    "can measure bias on real data."
                ),
                "reference_dataset_size":  n_tot,
                "reference_protected_cols": probe_protected,
                "reference_target_col":    probe_target,
                "cartography":             {},
                "constitution":            {},
                "model_biases":            [],
                "summary": {
                    "fair_score":             None,
                    "bias_count":             0,
                    "most_biased_attribute":  None,
                    "analysis_source":        "embedded_reference_dataset",
                    "model_type":             model_type,
                    "degenerate":             True,
                },
            }

        # ── 3. Bias Cartography on reference data + model predictions ────────
        carto_results = await cartography_service.run_cartography(
            dataset_csv=ref_csv,
            protected_cols=probe_protected,
            target_col=probe_target,
            model_predictions=model_predictions,
            audit_id=audit_id,
        )

        # ── 4. Counterfactual Constitution on reference data + model ─────────
        # Pass the FULL ref_df (including injected demographics) so constitution can
        # flip demographic columns.  Use only model feature cols for re-prediction.
        X_full = ref_df[[c for c in ref_df.columns if c != probe_target]]
        try:
            y_pred_arr = np.array(model_predictions)
            constitution_results = await constitution_service.generate_constitution(
                model=model,
                X=X_full,
                y_pred=y_pred_arr,
                protected_cols=probe_protected,
                feature_names=list(X_full.columns),
                cartography_results=carto_results,
                audit_id=audit_id,
            )
        except Exception as e:
            logger.warning(f"[{audit_id}] Constitution on reference dataset failed: {e}")
            constitution_results = {"error": str(e), "patterns": [], "sections": []}

        # ── 5. Extract structured bias list ──────────────────────────────────
        model_biases = self._extract_model_biases(carto_results, constitution_results)

        return {
            "audit_id":              audit_id,
            "analysis_type":         "model_probe",
            "reference_dataset_size": len(ref_df),
            "reference_protected_cols": probe_protected,
            "reference_target_col":  probe_target,
            "cartography":           carto_results,
            "constitution":          constitution_results,
            "model_biases":          model_biases,
            "summary": {
                "fair_score":             carto_results.get("fair_score", {}),
                "bias_count":             len(model_biases),
                "most_biased_attribute":  model_biases[0]["attribute"] if model_biases else None,
                "analysis_source":        "embedded_reference_dataset",
                "model_type":             model_type,
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_feature_names(model) -> Optional[List[str]]:
        """Extract feature names from sklearn-style model, including ensembles and pipelines."""
        raw = getattr(model, "_model", model)

        def _from_estimator(est) -> Optional[List[str]]:
            for attr in ("feature_names_in_", "feature_names"):
                val = getattr(est, attr, None)
                if val is not None:
                    return list(val)
            try:
                val = est.feature_name_()
                if val:
                    return list(val)
            except Exception:
                pass
            return None

        # Direct attributes
        found = _from_estimator(raw)
        if found:
            return found

        # Pipeline: check steps in reverse (last fitted step most likely has feature_names_in_)
        if hasattr(raw, "steps"):
            for _, step in reversed(raw.steps):
                found = _from_estimator(step)
                if found:
                    return found

        # VotingClassifier / StackingClassifier / BaggingClassifier
        for attr in ("estimators_", "estimators"):
            estimators = getattr(raw, attr, None)
            if not estimators:
                continue
            for est in estimators:
                est_raw = est[1] if isinstance(est, tuple) else est
                found = _from_estimator(est_raw)
                if found:
                    return found
                # Sub-estimator may also be a Pipeline
                if hasattr(est_raw, "steps"):
                    for _, step in reversed(est_raw.steps):
                        found = _from_estimator(step)
                        if found:
                            return found

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


model_probe_service = ModelBiasProbe()
