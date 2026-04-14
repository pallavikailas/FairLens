"""
Bias Cartography Service
========================
Generates a 2D 'fairness landscape' by:
1. Computing SHAP values for every prediction
2. Slicing by intersectional identity combinations (gender × race × geography, etc.)
3. Projecting decision residuals into 2D space using UMAP
4. Identifying bias hotspots (clusters with high residual disparity)

This surfaces WHERE in the feature space bias concentrates — not just a single number.
"""

import numpy as np
import pandas as pd
import shap
import umap
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Any, Optional
import logging

from app.core.config import settings
from app.services.gcp_client import bigquery_client, storage_client

logger = logging.getLogger(__name__)


class BiasCartographyService:
    """
    Produces intersectional bias topology maps.
    Core innovation: treats bias as a landscape, not a scalar.
    """

    PROTECTED_ATTRIBUTES = ["gender", "race", "age_group", "nationality", "religion", "disability"]

    def __init__(self):
        self.explainer = None
        self.umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=42
        )

    async def run_cartography(
        self,
        model: Any,
        X: pd.DataFrame,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray],
        protected_cols: List[str],
        audit_id: str,
    ) -> Dict[str, Any]:
        """
        Full cartography pipeline.
        Returns bias map coordinates + hotspot annotations.
        """
        logger.info(f"[{audit_id}] Starting bias cartography on {len(X)} samples")

        # Step 1: Compute SHAP values
        shap_values = self._compute_shap(model, X)

        # Step 2: Build intersectional slices
        slices = self._build_intersectional_slices(X, y_pred, y_true, protected_cols)

        # Step 3: UMAP projection of SHAP value space
        umap_coords = self._project_to_2d(shap_values, X)

        # Step 4: Compute per-point bias residual
        bias_residuals = self._compute_bias_residuals(X, y_pred, y_true, protected_cols)

        # Step 5: Identify hotspot clusters
        hotspots = self._identify_hotspots(umap_coords, bias_residuals, X, protected_cols)

        # Step 6: Compute aggregate fairness metrics per slice
        metrics = self._compute_slice_metrics(slices)

        # Step 7: Log to BigQuery
        await self._log_to_bigquery(audit_id, metrics, hotspots)

        result = {
            "audit_id": audit_id,
            "map_points": [
                {
                    "x": float(umap_coords[i, 0]),
                    "y": float(umap_coords[i, 1]),
                    "bias_score": float(bias_residuals[i]),
                    "slice_label": self._get_slice_label(X.iloc[i], protected_cols),
                    "prediction": int(y_pred[i]),
                    "ground_truth": int(y_true[i]) if y_true is not None else None,
                }
                for i in range(min(len(X), 2000))  # cap at 2k for frontend perf
            ],
            "hotspots": hotspots,
            "slice_metrics": metrics,
            "summary": {
                "total_samples": len(X),
                "hotspot_count": len(hotspots),
                "most_biased_slice": max(metrics, key=lambda m: m["bias_magnitude"]) if metrics else None,
                "overall_bias_score": float(np.mean(np.abs(bias_residuals))),
            }
        }

        logger.info(f"[{audit_id}] Cartography complete. {len(hotspots)} hotspots identified.")
        return result

    def _compute_shap(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values. Handles tree models (TreeExplainer) and black-boxes (KernelExplainer)."""
        try:
            self.explainer = shap.TreeExplainer(model)
            shap_vals = self.explainer.shap_values(X)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # binary classification: take positive class
        except Exception:
            # Fallback: KernelExplainer for non-tree models (slower)
            background = shap.sample(X, min(100, len(X)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = self.explainer.shap_values(X.iloc[:500])  # sample for speed
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        return np.array(shap_vals)

    def _build_intersectional_slices(
        self, X: pd.DataFrame, y_pred: np.ndarray,
        y_true: Optional[np.ndarray], protected_cols: List[str]
    ) -> List[Dict]:
        """Create all intersectional subgroup slices."""
        present_cols = [c for c in protected_cols if c in X.columns]
        if not present_cols:
            return []

        slices = []
        # Single-attribute slices
        for col in present_cols:
            for val in X[col].unique():
                mask = X[col] == val
                if mask.sum() < 10:
                    continue
                slices.append({
                    "label": f"{col}={val}",
                    "type": "single",
                    "attributes": {col: val},
                    "indices": np.where(mask)[0].tolist(),
                    "size": int(mask.sum()),
                    "pred_positive_rate": float(y_pred[mask].mean()),
                    "true_positive_rate": float(y_true[mask].mean()) if y_true is not None else None,
                })

        # Pairwise intersectional slices (the key innovation)
        if len(present_cols) >= 2:
            for i, col1 in enumerate(present_cols):
                for col2 in present_cols[i+1:]:
                    for v1 in X[col1].unique():
                        for v2 in X[col2].unique():
                            mask = (X[col1] == v1) & (X[col2] == v2)
                            if mask.sum() < 10:
                                continue
                            slices.append({
                                "label": f"{col1}={v1} ∩ {col2}={v2}",
                                "type": "intersectional",
                                "attributes": {col1: v1, col2: v2},
                                "indices": np.where(mask)[0].tolist(),
                                "size": int(mask.sum()),
                                "pred_positive_rate": float(y_pred[mask].mean()),
                                "true_positive_rate": float(y_true[mask].mean()) if y_true is not None else None,
                            })
        return slices

    def _project_to_2d(self, shap_values: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """UMAP projection of SHAP value space into 2D."""
        n = min(len(shap_values), 5000)
        return self.umap_reducer.fit_transform(shap_values[:n])

    def _compute_bias_residuals(
        self, X: pd.DataFrame, y_pred: np.ndarray,
        y_true: Optional[np.ndarray], protected_cols: List[str]
    ) -> np.ndarray:
        """
        Per-sample bias score: deviation from group-conditional mean prediction.
        High residual = this sample's prediction diverges from what the model
        predicts for similar demographics.
        """
        residuals = np.zeros(len(X))
        present_cols = [c for c in protected_cols if c in X.columns]
        if not present_cols:
            return residuals

        # Use the first available protected attribute for grouping
        col = present_cols[0]
        for val in X[col].unique():
            mask = X[col] == val
            if mask.sum() == 0:
                continue
            group_mean = y_pred[mask].mean()
            overall_mean = y_pred.mean()
            residuals[mask] = np.abs(y_pred[mask] - group_mean) + abs(group_mean - overall_mean)

        return residuals

    def _identify_hotspots(
        self, coords: np.ndarray, residuals: np.ndarray,
        X: pd.DataFrame, protected_cols: List[str]
    ) -> List[Dict]:
        """Identify bias hotspot clusters in UMAP space (top-10 by bias magnitude)."""
        # Grid-based density of high-residual points
        hotspots = []
        threshold = np.percentile(residuals, 85)
        high_bias_mask = residuals > threshold
        high_bias_coords = coords[high_bias_mask]

        if len(high_bias_coords) == 0:
            return hotspots

        # Simple grid clustering
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(high_bias_coords)

        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:
                continue
            cluster_mask = clustering.labels_ == cluster_id
            cluster_indices = np.where(high_bias_mask)[0][cluster_mask]

            # Dominant identity slice in this cluster
            dominant_slice = "unknown"
            present_cols = [c for c in protected_cols if c in X.columns]
            if present_cols and len(cluster_indices) > 0:
                col = present_cols[0]
                slice_counts = X.iloc[cluster_indices][col].value_counts()
                if len(slice_counts) > 0:
                    dominant_slice = f"{col}={slice_counts.index[0]}"

            hotspots.append({
                "cluster_id": int(cluster_id),
                "centroid_x": float(high_bias_coords[cluster_mask][:, 0].mean()),
                "centroid_y": float(high_bias_coords[cluster_mask][:, 1].mean()),
                "size": int(cluster_mask.sum()),
                "mean_bias_magnitude": float(residuals[cluster_indices].mean()),
                "dominant_slice": dominant_slice,
                "severity": "high" if residuals[cluster_indices].mean() > np.percentile(residuals, 95) else "medium",
            })

        return sorted(hotspots, key=lambda h: h["mean_bias_magnitude"], reverse=True)[:10]

    def _compute_slice_metrics(self, slices: List[Dict]) -> List[Dict]:
        """Compute fairness metrics per slice."""
        if not slices:
            return []

        overall_rate = np.mean([s["pred_positive_rate"] for s in slices])
        metrics = []
        for s in slices:
            rate = s["pred_positive_rate"]
            spd = rate - overall_rate  # Statistical Parity Difference
            di = rate / overall_rate if overall_rate > 0 else 0  # Disparate Impact
            metrics.append({
                **s,
                "statistical_parity_diff": round(spd, 4),
                "disparate_impact": round(di, 4),
                "bias_magnitude": round(abs(spd), 4),
                "flagged": abs(spd) > settings.DEMOGRAPHIC_PARITY_THRESHOLD or di < settings.DISPARATE_IMPACT_THRESHOLD,
            })

        return sorted(metrics, key=lambda m: m["bias_magnitude"], reverse=True)

    def _get_slice_label(self, row: pd.Series, protected_cols: List[str]) -> str:
        parts = [f"{c}={row[c]}" for c in protected_cols if c in row.index]
        return " | ".join(parts) if parts else "unknown"

    async def _log_to_bigquery(self, audit_id: str, metrics: List[Dict], hotspots: List[Dict]):
        """Log audit results to BigQuery for compliance trail."""
        try:
            rows = [{
                "audit_id": audit_id,
                "stage": "cartography",
                "hotspot_count": len(hotspots),
                "flagged_slice_count": sum(1 for m in metrics if m.get("flagged")),
                "timestamp": pd.Timestamp.now().isoformat(),
            }]
            await bigquery_client.insert_rows(settings.BIGQUERY_TABLE_AUDITS, rows)
        except Exception as e:
            logger.warning(f"BigQuery log failed (non-fatal): {e}")


cartography_service = BiasCartographyService()
