"""
API routes for Bias Cartography.
Accepts models via:
  - .pkl upload (sklearn / XGBoost / LightGBM / CatBoost)
  - model_type=pytorch + .pt file
  - model_type=api + endpoint URL
  - model_type=vertex_ai + endpoint_id
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import pickle
import io
import uuid
from typing import Optional

from app.services.cartography import cartography_service
from app.services.model_adapter import FairLensAdapter, BaseModelAdapter

router = APIRouter()


def _load_adapter(
    model_type: str,
    model_file,
    api_endpoint: Optional[str],
    vertex_endpoint_id: Optional[str],
    gcp_project: Optional[str],
) -> BaseModelAdapter:
    t = (model_type or "sklearn").lower()

    if t in ("sklearn", "xgboost", "lightgbm", "catboost", "pickle"):
        if not model_file:
            raise HTTPException(400, "model_file required for sklearn model_type")
        model_bytes = model_file.file.read()
        model = pickle.loads(model_bytes)
        return FairLensAdapter.from_sklearn(model)

    elif t == "api":
        if not api_endpoint:
            raise HTTPException(400, "api_endpoint required for REST model_type")
        return FairLensAdapter.from_api(api_endpoint)

    elif t == "huggingface":
        name = api_endpoint or "distilbert-base-uncased-finetuned-sst-2-english"
        return FairLensAdapter.from_huggingface(name)

    elif t in ("vertex_ai", "vertexai"):
        if not vertex_endpoint_id or not gcp_project:
            raise HTTPException(400, "vertex_endpoint_id and gcp_project required")
        return FairLensAdapter.from_vertex_ai(vertex_endpoint_id, project=gcp_project)

    else:
        raise HTTPException(400, f"Unknown model_type '{t}'. Supported: sklearn, api, huggingface, vertex_ai")


@router.post("/analyze")
async def analyze_bias_cartography(
    dataset_file: UploadFile = File(...),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    model_type: str = Form(default="sklearn"),
    model_file: Optional[UploadFile] = File(default=None),
    api_endpoint: Optional[str] = Form(default=None),
    vertex_endpoint_id: Optional[str] = Form(default=None),
    gcp_project: Optional[str] = Form(default=None),
):
    """
    Stage 1: Bias Cartography — works with ANY model type.

    Supported model_type values:
    - sklearn  (upload .pkl)
    - api      (provide api_endpoint URL)
    - huggingface (provide model name in api_endpoint)
    - vertex_ai (provide vertex_endpoint_id + gcp_project)
    """
    audit_id = str(uuid.uuid4())[:8]
    try:
        csv_bytes = await dataset_file.read()
        df = pd.read_csv(io.BytesIO(csv_bytes))
        protected = [c.strip() for c in protected_cols.split(",")]
        X = df[[c for c in df.columns if c != target_col]]
        y_true = df[target_col].values if target_col in df.columns else None

        adapter = _load_adapter(model_type, model_file, api_endpoint, vertex_endpoint_id, gcp_project)
        y_pred = adapter.predict(X)

        result = await cartography_service.run_cartography(
            model=adapter, X=X, y_pred=y_pred, y_true=y_true,
            protected_cols=protected, audit_id=audit_id,
        )
        result["model_type"] = adapter.get_model_type()
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cartography failed: {str(e)}")
