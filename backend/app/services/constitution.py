"""
Counterfactual Constitution Service
=====================================
Uses Gemini to generate a structured 'constitution' document that captures
what the model would have decided if demographic attributes were different.

Key innovation:
- Not just individual counterfactuals, but a VERSIONED DOCUMENT that captures
  the model's implicit decision policy across demographic axes
- Diffs the constitution across model versions to detect fairness drift
- Human-readable enough for legal/HR teams, precise enough for engineers
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

from app.services.gemini_client import ask_gemini
from app.core.config import settings

logger = logging.getLogger(__name__)


class CounterfactualConstitutionService:
    """
    Generates the Counterfactual Constitution via Gemini.
    The constitution answers: "What implicit rules is this model following,
    and how do those rules change when demographics change?"
    Uses the central gemini_client (Vertex AI via ADC).
    """

    async def generate_constitution(
        self,
        model: Optional[Any],
        X: pd.DataFrame,
        y_pred: np.ndarray,
        protected_cols: List[str],
        feature_names: List[str],
        cartography_results: Dict,
        audit_id: str,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Generate counterfactual pairs for each protected attribute (skipped if no model)
        2. Aggregate patterns into policy statements
        3. Ask Gemini to synthesise into a structured Constitution document
        4. Return structured JSON + human-readable Markdown
        """
        logger.info(f"[{audit_id}] Generating Counterfactual Constitution (model={'provided' if model else 'none'})")

        # API-based models are slow — limit CF samples to avoid Cloud Run timeouts.
        # HF classifiers: ~0.3s/call × 200 samples × 3 attrs × 2 flips = 360s (too slow).
        # LLMs are even slower. Local sklearn models can handle 200 comfortably.
        model_type_str = (model.get_model_type() if hasattr(model, "get_model_type") else "") or ""
        is_generative = "GenerativeLLM" in model_type_str
        is_api_model = is_generative or any(t in model_type_str for t in ("HuggingFace", "REST:"))
        cf_n_samples = 20 if is_generative else 50 if is_api_model else 200

        # Step 1: Build counterfactual pairs (empty when no model available)
        cf_pairs = self._generate_cf_pairs(model, X, y_pred, protected_cols, n_samples=cf_n_samples)

        # Step 2: Extract decision patterns
        patterns = self._extract_patterns(cf_pairs, protected_cols)

        # Step 3: Gemini synthesis
        constitution_text = await self._gemini_synthesise(
            patterns, cf_pairs, cartography_results, protected_cols, feature_names, audit_id,
            model_available=(model is not None),
            model=model,
        )

        # Step 4: Parse into structured sections
        sections = self._parse_constitution(constitution_text)

        result = {
            "audit_id": audit_id,
            "generated_at": datetime.utcnow().isoformat(),
            "constitution_markdown": constitution_text,
            "sections": sections,
            "counterfactual_pairs": cf_pairs[:50],  # sample for frontend
            "patterns": patterns,
            "summary": {
                "total_cf_pairs": len(cf_pairs),
                "decision_flips": sum(1 for p in cf_pairs if p["decision_flipped"]),
                "flip_rate": round(sum(1 for p in cf_pairs if p["decision_flipped"]) / max(len(cf_pairs), 1), 3),
                "most_sensitive_attribute": max(
                    protected_cols,
                    key=lambda c: sum(1 for p in cf_pairs if p["changed_attr"] == c and p["decision_flipped"]),
                    default=None
                ) if protected_cols else None,
            }
        }

        logger.info(f"[{audit_id}] Constitution generated. Flip rate: {result['summary']['flip_rate']:.1%}")
        return result

    def _generate_cf_pairs(
        self, model: Optional[Any], X: pd.DataFrame, y_pred: np.ndarray,
        protected_cols: List[str], n_samples: int = 200
    ) -> List[Dict]:
        """
        For each sample, flip each protected attribute to all other observed values,
        re-predict, and record whether the decision changed.
        Returns empty list when no model is available.
        """
        if model is None:
            return []

        pairs = []
        present_cols = [c for c in protected_cols if c in X.columns]
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)

        for idx in indices:
            if idx >= len(y_pred):
                continue  # guard against predict returning fewer rows than X
            row = X.iloc[idx].copy()
            original_pred = y_pred[idx]

            for col in present_cols:
                original_val = row[col]
                other_vals = [v for v in X[col].unique() if v != original_val][:3]
                if not other_vals:
                    continue

                # Batch original + all counterfactuals so the encoder sees ALL values
                # for this column at once — prevents fresh single-row LabelEncoders
                # from mapping every value to 0 (making flips invisible to the model).
                row_dict = row.to_dict()
                cf_rows = []
                for v in other_vals:
                    cf = row_dict.copy()
                    cf[col] = v
                    cf_rows.append(cf)
                batch_df = pd.DataFrame([row_dict] + cf_rows)

                try:
                    batch_preds = model.predict(batch_df)
                    has_proba = hasattr(model, "predict_proba")
                    batch_probs = model.predict_proba(batch_df)[:, 1] if has_proba else None
                except Exception:
                    continue

                orig_prob = float(batch_probs[0]) if batch_probs is not None else None
                for i, alt_val in enumerate(other_vals):
                    cf_pred = batch_preds[i + 1]
                    cf_prob = float(batch_probs[i + 1]) if batch_probs is not None else None
                    pairs.append({
                        "sample_idx": int(idx),
                        "changed_attr": col,
                        "original_value": str(original_val),
                        "counterfactual_value": str(alt_val),
                        "original_prediction": int(original_pred),
                        "counterfactual_prediction": int(cf_pred),
                        "original_prob": orig_prob,
                        "counterfactual_prob": cf_prob,
                        "decision_flipped": int(original_pred) != int(cf_pred),
                        "prob_delta": float(cf_prob - orig_prob) if cf_prob is not None and orig_prob is not None else None,
                    })

        return pairs

    def _extract_patterns(self, cf_pairs: List[Dict], protected_cols: List[str]) -> List[Dict]:
        """Aggregate counterfactual pairs into pattern statements."""
        patterns = []
        for col in protected_cols:
            col_pairs = [p for p in cf_pairs if p["changed_attr"] == col]
            if not col_pairs:
                continue

            flip_rate = sum(1 for p in col_pairs if p["decision_flipped"]) / max(len(col_pairs), 1)
            avg_prob_delta = np.mean([abs(p["prob_delta"]) for p in col_pairs if p["prob_delta"] is not None])

            # Direction of bias: which value benefits?
            value_flip_rates = {}
            for val in set(p["original_value"] for p in col_pairs):
                val_pairs = [p for p in col_pairs if p["original_value"] == val]
                if val_pairs:
                    value_flip_rates[val] = sum(1 for p in val_pairs if p["decision_flipped"]) / len(val_pairs)

            patterns.append({
                "attribute": col,
                "flip_rate": round(flip_rate, 3),
                "avg_probability_shift": round(float(avg_prob_delta), 3) if not np.isnan(avg_prob_delta) else 0,
                "flip_rate_by_value": value_flip_rates,
                "bias_direction": max(value_flip_rates, key=value_flip_rates.get) if value_flip_rates else "unknown",
                "severity": "critical" if flip_rate > 0.3 else "high" if flip_rate > 0.15 else "medium" if flip_rate > 0.05 else "low",
            })

        return sorted(patterns, key=lambda p: p["flip_rate"], reverse=True)

    async def _gemini_synthesise(
        self,
        patterns: List[Dict],
        cf_pairs: List[Dict],
        cartography_results: Dict,
        protected_cols: List[str],
        feature_names: List[str],
        audit_id: str,
        model_available: bool = True,
        model: Optional[Any] = None,
    ) -> str:
        """Use Gemini to synthesise patterns + counterfactuals into a Constitution document."""
        hotspots_summary = json.dumps(
            cartography_results.get("hotspots", [])[:3], indent=2
        )
        patterns_summary = json.dumps(patterns, indent=2) if patterns else "No counterfactual flip patterns detected — the model may be insensitive to the protected attributes, or all samples had the same attribute value."
        flip_examples = json.dumps(
            [p for p in cf_pairs if p.get("decision_flipped")][:10], indent=2
        ) if cf_pairs else "No decision flips detected across counterfactual pairs."

        if not model_available:
            model_note = "\nNOTE: No ML model was provided. Analysis is based on dataset statistics and the bias topology map only. Counterfactual simulation is not available.\n"
        else:
            model_name = getattr(model, "_name", None) or getattr(model, "get_model_type", lambda: "")()
            if "AutoReference" in str(model_name):
                model_note = "\nNOTE: No user-provided model was uploaded. FairLens auto-trained a Logistic Regression reference model on this dataset to enable counterfactual simulation. The counterfactual results reveal data-level bias — the implicit patterns any model would learn from this training data.\n"
            else:
                model_note = ""

        prompt = f"""You are an AI fairness auditor generating a "Counterfactual Constitution" —
a structured policy document that reveals the IMPLICIT RULES an AI model is following
with respect to demographic attributes.

AUDIT ID: {audit_id}
PROTECTED ATTRIBUTES ANALYZED: {', '.join(protected_cols)}
FEATURES IN MODEL: {', '.join(feature_names[:20])}
{model_note}
COUNTERFACTUAL PATTERNS (what changes when demographics change):
{patterns_summary}

DECISION FLIP EXAMPLES (real cases where demographic change flipped the outcome):
{flip_examples}

BIAS HOTSPOT SUMMARY (from topology map):
{hotspots_summary}

Generate a Counterfactual Constitution document with exactly these sections:

## 1. Executive Summary
(2-3 sentences: what did we find, how serious is it, who is affected)

## 2. Implicit Decision Rules
(List the model's apparent rules as plain-English "IF...THEN" statements derived from counterfactual evidence.
E.g. "IF applicant is male THEN approval probability increases by ~{{X}}% independent of qualifications")

## 3. Demographic Sensitivity Index
(For each protected attribute: flip rate, probability shift, severity, and plain-English explanation
of what the flip means in the real world — job denied, loan rejected, etc.)

## 4. Most Affected Groups
(Rank the demographic intersections most disadvantaged by the model's decisions)

## 5. Structural vs. Proxy Bias
(Is the bias direct — the model directly uses demographics — or indirect via proxy features?)

## 6. Legal Risk Assessment
(Under EU AI Act, US EEOC, and the 4/5ths rule for disparate impact)

## 7. Recommended Remediation Priority
(Which bias should be fixed first, and why, based on impact severity)

Write in clear, accessible language. Use specific numbers from the data. 
Make it readable for both a non-technical HR manager and a data scientist.
Use markdown formatting."""

        return await ask_gemini(prompt)

    def _parse_constitution(self, markdown_text: str) -> List[Dict]:
        """Parse markdown sections into structured JSON."""
        sections = []
        current_section = None

        for line in markdown_text.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "title": line.replace("## ", "").strip(),
                    "content": ""
                }
            elif current_section:
                current_section["content"] += line + "\n"

        if current_section:
            sections.append(current_section)

        return sections


constitution_service = CounterfactualConstitutionService()
