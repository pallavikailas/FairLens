"""API routes for Counterfactual Constitution."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle, io, uuid, json
from sklearn.preprocessing import LabelEncoder

from app.services.constitution import constitution_service
from app.services.auto_detect import auto_detect_columns
from app.services.dataset_loader import load_dataset_csv

router = APIRouter()


def _resolve_feature_cols(model, df: pd.DataFrame, fallback_target: str) -> List[str]:
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        if names and all(n in df.columns for n in names):
            return names
    if hasattr(model, "feature_names") and model.feature_names:
        names = model.feature_names
        if all(n in df.columns for n in names):
            return list(names)
    if hasattr(model, "feature_name_"):
        try:
            names = model.feature_name_()
            if names and all(n in df.columns for n in names):
                return list(names)
        except Exception:
            pass
    return [c for c in df.columns if c != fallback_target]




@router.post("/generate")
async def generate_constitution(
    dataset_file: Optional[UploadFile] = File(default=None),
    model_file: Optional[UploadFile] = File(default=None),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    cartography_results: str = Form(..., description="JSON string from cartography stage"),
    dataset_source: str = Form(default="upload"),
    dataset_url: str = Form(default=""),
):
    audit_id = str(uuid.uuid4())[:8]
    try:
        # Load dataset
        dataset_csv = await load_dataset_csv(dataset_file, dataset_source, dataset_url)
        df = pd.read_csv(io.StringIO(dataset_csv))

        # Resolve protected_cols — handle 'auto' sentinel
        is_auto = protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(",")
        if is_auto:
            detected = await auto_detect_columns(dataset_csv, audit_id)
            protected = detected["protected_cols"]
            tgt = detected["target_col"]
        else:
            protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]
            tgt = target_col if target_col and target_col != "auto" else None

        # Fallback target column detection
        if not tgt or tgt not in df.columns:
            for col in df.columns:
                if df[col].nunique() == 2:
                    tgt = col
                    break
            else:
                tgt = df.columns[-1]

        # Load model if provided
        model = None
        if model_file is not None:
            model_bytes = await model_file.read()
            if model_bytes:
                try:
                    model = pickle.loads(model_bytes)
                except Exception:
                    try:
                        import joblib
                        model = joblib.load(io.BytesIO(model_bytes))
                    except Exception as e:
                        raise HTTPException(400, f"Failed to load model file: {e}")

        feature_cols = _resolve_feature_cols(model, df, tgt) if model is not None else [c for c in df.columns if c != tgt]
        X = df[feature_cols]

        # Generate predictions only when model is available
        if model is not None:
            try:
                y_pred = model.predict(X)
            except Exception:
                # Fallback: label-encode categorical columns (needed for XGBoost/LightGBM)
                try:
                    X_enc = X.copy()
                    le = LabelEncoder()
                    for col in X_enc.select_dtypes(include=["object", "category"]).columns:
                        try:
                            X_enc[col] = le.fit_transform(X_enc[col].astype(str))
                        except Exception:
                            X_enc[col] = 0
                    y_pred = model.predict(X_enc.fillna(0))
                except Exception as e:
                    raise HTTPException(500, f"Model prediction failed: {e}")
        else:
            # Dataset-only mode: use target column values as pseudo-predictions
            y_pred = df[tgt].values if tgt in df.columns else np.zeros(len(df), dtype=int)

        carto = json.loads(cartography_results)

        result = await constitution_service.generate_constitution(
            model=model,
            X=X,
            y_pred=y_pred,
            protected_cols=protected,
            feature_names=feature_cols,
            cartography_results=carto,
            audit_id=audit_id,
        )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
