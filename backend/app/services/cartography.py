"""
Bias Cartography Service — Cloud-Native (Gemini-powered)
=========================================================
No SHAP. No UMAP.

1. Computes statistical bias metrics per demographic slice
2. Identifies intersectional bias patterns
3. Generates 2D topology coordinates via bias score + prediction scatter
4. Returns hotspot clusters with plain-English explanations via Gemini
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional

from app.services.gemini_client import ask_gemini_json
from app.core.config import settings

logger = logging.getLogger(__name__)


class BiasCartographyService:

    async def run_cartography(
        self,
        dataset_csv: str,
        protected_cols: List[str],
        target_col: str,
        model_predictions: Optional[List] = None,
        audit_id: str = "",
    ) -> Dict[str, Any]:

        logger.info(f"[{audit_id}] Starting cloud Bias Cartography")

        import io
        df = pd.read_csv(io.StringIO(dataset_csv))
        sample_df = df.sample(min(300, len(df)), random_state=42)

        slice_metrics = self._compute_slice_metrics(df, protected_cols, target_col, model_predictions)
        gemini_analysis = await self._gemini_analyse(sample_df, protected_cols, target_col, slice_metrics, audit_id, model_predictions is not None)
        map_points = self._generate_map_points(df, protected_cols, target_col, slice_metrics, model_predictions)
        hotspots = self._identify_hotspots(slice_metrics)

        return {
            "audit_id": audit_id,
            "map_points": map_points,
            "hotspots": hotspots,
            "slice_metrics": slice_metrics,
            "gemini_analysis": gemini_analysis,
            "summary": {
                "total_samples": len(df),
                "hotspot_count": len(hotspots),
                "protected_cols_found": [c for c in protected_cols if c in df.columns],
                "overall_bias_score": round(
                    np.mean([abs(m["statistical_parity_diff"]) for m in slice_metrics]) if slice_metrics else 0, 3
                ),
                "most_biased_slice": slice_metrics[0]["label"] if slice_metrics else None,
            }
        }

    def _compute_slice_metrics(self, df, protected_cols, target_col, model_predictions=None):
        present = [c for c in protected_cols if c in df.columns]
        if not present or target_col not in df.columns:
            return []
        if model_predictions is not None and len(model_predictions) == len(df):
            target = pd.Series(model_predictions, index=df.index, dtype=float)
        else:
            target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        overall_rate = float(target.mean())
        if overall_rate == 0:
            return []
        metrics = []
        for col in present:
            for val in df[col].dropna().unique():
                mask = df[col] == val
                if mask.sum() < 5:
                    continue
                group_rate = float(target[mask].mean())
                spd = round(group_rate - overall_rate, 4)
                di = round(group_rate / overall_rate, 4) if overall_rate > 0 else 0
                metrics.append({
                    "label": f"{col}={val}", "attribute": col, "value": str(val),
                    "size": int(mask.sum()), "positive_rate": round(group_rate, 4),
                    "overall_rate": round(overall_rate, 4),
                    "statistical_parity_diff": spd, "disparate_impact": di,
                    "bias_magnitude": round(abs(spd), 4),
                    "flagged": abs(spd) > settings.DEMOGRAPHIC_PARITY_THRESHOLD or di < settings.DISPARATE_IMPACT_THRESHOLD,
                })
        if len(present) >= 2:
            for i, c1 in enumerate(present):
                for c2 in present[i+1:]:
                    for v1 in df[c1].dropna().unique()[:4]:
                        for v2 in df[c2].dropna().unique()[:4]:
                            mask = (df[c1] == v1) & (df[c2] == v2)
                            if mask.sum() < 5:
                                continue
                            group_rate = float(target[mask].mean())
                            spd = round(group_rate - overall_rate, 4)
                            di = round(group_rate / overall_rate, 4) if overall_rate > 0 else 0
                            metrics.append({
                                "label": f"{c1}={v1} ∩ {c2}={v2}", "attribute": f"{c1}+{c2}",
                                "value": f"{v1}+{v2}", "size": int(mask.sum()),
                                "positive_rate": round(group_rate, 4), "overall_rate": round(overall_rate, 4),
                                "statistical_parity_diff": spd, "disparate_impact": di,
                                "bias_magnitude": round(abs(spd), 4),
                                "flagged": abs(spd) > settings.DEMOGRAPHIC_PARITY_THRESHOLD or di < settings.DISPARATE_IMPACT_THRESHOLD,
                            })
        return sorted(metrics, key=lambda m: m["bias_magnitude"], reverse=True)

    async def _gemini_analyse(self, df, protected_cols, target_col, slice_metrics, audit_id, using_model_predictions=False):
        top_slices = json.dumps(slice_metrics[:10], indent=2)
        col_summary = {col: df[col].value_counts().head(8).to_dict() for col in protected_cols if col in df.columns}
        analysis_source = "model prediction outputs (what the uploaded model actually decides)" if using_model_predictions else "dataset ground-truth labels"
        prompt = f"""You are an AI fairness auditor analysing a model for bias.
ANALYSIS SOURCE: {analysis_source}
DATASET: {len(df)} rows, target='{target_col}', protected={protected_cols}
DISTRIBUTIONS: {json.dumps(col_summary)}
TOP BIAS FINDINGS: {top_slices}
Return ONLY this JSON:
{{"severity":"critical|high|medium|low","headline":"one sentence","key_findings":["f1","f2","f3"],"most_affected_group":"group","bias_type":"direct|proxy|intersectional|systemic","real_world_impact":"impact","legal_risk":"risk","recommended_action":"action"}}"""
        try:
            return await ask_gemini_json(prompt)
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}")
            return {"severity": "unknown", "headline": "Analysis unavailable", "key_findings": []}

    def _generate_map_points(self, df, protected_cols, target_col, slice_metrics, model_predictions=None):
        present = [c for c in protected_cols if c in df.columns]
        if model_predictions is not None and len(model_predictions) == len(df):
            target = pd.Series(model_predictions, index=df.index, dtype=float)
        elif target_col in df.columns:
            target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        else:
            target = pd.Series([0] * len(df))
        bias_lookup = {m["label"]: m["bias_magnitude"] for m in slice_metrics}
        points = []
        sample = df.sample(min(500, len(df)), random_state=42)
        for idx, row in sample.iterrows():
            bias_score = max((bias_lookup.get(f"{col}={row[col]}", 0.0) for col in present if col in row), default=0.0)
            pred_val = float(target.loc[idx]) if idx in target.index else 0.0
            points.append({
                "x": round(float(bias_score) + np.random.normal(0, 0.02), 4),
                "y": round(pred_val + np.random.normal(0, 0.05), 4),
                "bias_score": round(bias_score, 4),
                "slice_label": " | ".join(f"{col}={row[col]}" for col in present if col in row) or "unknown",
                "prediction": int(pred_val),
            })
        return points

    def _identify_hotspots(self, slice_metrics):
        return [
            {
                "cluster_id": i, "centroid_x": m["bias_magnitude"], "centroid_y": m["positive_rate"],
                "size": m["size"], "mean_bias_magnitude": m["bias_magnitude"],
                "dominant_slice": m["label"],
                "severity": "critical" if m["bias_magnitude"] > 0.3 else "high" if m["bias_magnitude"] > 0.15 else "medium",
                "statistical_parity_diff": m["statistical_parity_diff"], "disparate_impact": m["disparate_impact"],
            }
            for i, m in enumerate([m for m in slice_metrics if m.get("flagged")][:8])
        ]


cartography_service = BiasCartographyService()
