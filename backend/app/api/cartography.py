"""
Cartography API — bridges the cloud-native frontend with the original SHAP/UMAP service.
Accepts CSV (upload or online source), auto-detects columns, runs the full pipeline.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uuid, io, httpx
import pandas as pd
import numpy as np

from app.services.cartography import cartography_service
from app.services.auto_detect import auto_detect_columns

router = APIRouter()


async def _load_csv(
    dataset_file: Optional[UploadFile],
    dataset_source: str,
    dataset_url: str,
) -> pd.DataFrame:
    """Load dataset from any source into a DataFrame."""

    if dataset_source == 'upload' or not dataset_source or not dataset_url:
        if not dataset_file:
            raise HTTPException(400, "dataset_file required")
        raw = await dataset_file.read()
        return pd.read_csv(io.BytesIO(raw))

    elif dataset_source == 'url' and dataset_url:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(dataset_url)
            resp.raise_for_status()
            return pd.read_csv(io.StringIO(resp.text))

    elif dataset_source == 'huggingface' and dataset_url:
        name = dataset_url.strip()
        async with httpx.AsyncClient(timeout=30) as client:

            # Step 1: discover available configs and splits
            info_resp = await client.get(
                f"https://datasets-server.huggingface.co/info?dataset={name}"
            )
            config = "default"
            split = "train"
            if info_resp.status_code == 200:
                info = info_resp.json()
                configs = list(info.get("dataset_info", {}).keys())
                if configs:
                    config = configs[0]
                splits = list(info.get("dataset_info", {}).get(config, {}).get("splits", {}).keys())
                if splits:
                    split = splits[0]

            # Step 2: fetch rows
            rows_resp = await client.get(
                f"https://datasets-server.huggingface.co/rows"
                f"?dataset={name}&config={config}&split={split}&offset=0&length=300"
            )
            if rows_resp.status_code != 200:
                raise HTTPException(400,
                    f"Could not load HuggingFace dataset '{name}'. "
                    f"Status: {rows_resp.status_code}. "
                    f"Try downloading the CSV directly from huggingface.co/datasets/{name} and uploading it."
                )
            data = rows_resp.json()
            rows = data.get("rows", [])
            if not rows:
                raise HTTPException(400, f"HuggingFace dataset '{name}' returned no rows.")
            return pd.DataFrame([r["row"] for r in rows])

    elif dataset_source == 'kaggle':
        raise HTTPException(400,
            "Kaggle requires authentication. Please download the CSV from Kaggle and upload it directly.")

    raise HTTPException(400, "No valid dataset provided")


class _DatasetOnlyModel:
    """
    Dummy model used when no model file is uploaded.
    Uses the target column's values as pseudo-predictions so the
    cartography service can still compute bias metrics.
    """
    def __init__(self, y: np.ndarray):
        self._y = y

    def predict(self, X):
        return self._y[:len(X)]

    def predict_proba(self, X):
        p = self._y[:len(X)].astype(float)
        return np.column_stack([1 - p, p])


@router.post("/analyze")
async def analyze_bias_cartography(
    dataset_file: Optional[UploadFile] = File(default=None),
    model_file: Optional[UploadFile] = File(default=None),
    protected_cols: str = Form(default="auto"),
    target_col: str = Form(default="auto"),
    model_type: str = Form(default="sklearn"),
    api_endpoint: str = Form(default=""),
    vertex_endpoint_id: str = Form(default=""),
    gcp_project: str = Form(default=""),
    dataset_source: str = Form(default="upload"),
    dataset_url: str = Form(default=""),
):
    audit_id = str(uuid.uuid4())[:8]
    try:
        # 1. Load dataset
        df = await _load_csv(dataset_file, dataset_source, dataset_url)

        # 2. Auto-detect or use provided columns
        csv_str = df.to_csv(index=False)
        if protected_cols in ("auto", "", "['auto']", "auto,"):
            detected = await auto_detect_columns(csv_str, audit_id)
            protected = detected["protected_cols"]
            target = detected["target_col"]
        else:
            protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]
            target = target_col

        # Fallback if detection failed
        if not protected:
            protected = []
        if not target or target not in df.columns:
            # Pick first binary column
            for col in df.columns:
                if df[col].nunique() == 2:
                    target = col
                    break
            else:
                target = df.columns[-1]

        # 3. Load model or use dummy
        feature_cols = [c for c in df.columns if c != target]
        X = df[feature_cols].copy()
        y_true = pd.to_numeric(df[target], errors="coerce").fillna(0).values

        if model_file and model_file.filename:
            import pickle
            model_bytes = await model_file.read()
            model = pickle.loads(model_bytes)
            # Encode categoricals for sklearn
            from sklearn.preprocessing import LabelEncoder
            X_enc = X.copy()
            for col in X_enc.select_dtypes(include=["object", "category"]).columns:
                try:
                    X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
                except Exception:
                    X_enc[col] = 0
            X_enc = X_enc.fillna(0)
            y_pred = model.predict(X_enc)
        else:
            # No model — use target column as predictions
            model = _DatasetOnlyModel(y_true)
            y_pred = y_true.copy()

        # 4. Run cartography
        result = await cartography_service.run_cartography(
            model=model,
            X=X,
            y_pred=y_pred,
            y_true=y_true,
            protected_cols=protected,
            audit_id=audit_id,
        )

        result["detected_protected_cols"] = protected
        result["detected_target_col"] = target
        result["model_type"] = model_type
        result["dataset_source"] = dataset_source

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cartography failed: {str(e)}")
