"""API routes for Counterfactual Constitution."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import pickle, io, uuid, json

from app.services.constitution import constitution_service

router = APIRouter()

@router.post("/generate")
async def generate_constitution(
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    cartography_results: str = Form(..., description="JSON string from cartography stage"),
):
    audit_id = str(uuid.uuid4())[:8]
    try:
        model = pickle.loads(await model_file.read())
        df = pd.read_csv(io.BytesIO(await dataset_file.read()))
        protected = [c.strip() for c in protected_cols.split(",")]
        X = df[[c for c in df.columns if c != target_col]]
        y_pred = model.predict(X)
        carto = json.loads(cartography_results)

        result = await constitution_service.generate_constitution(
            model=model, X=X, y_pred=y_pred,
            protected_cols=protected,
            feature_names=X.columns.tolist(),
            cartography_results=carto,
            audit_id=audit_id,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
