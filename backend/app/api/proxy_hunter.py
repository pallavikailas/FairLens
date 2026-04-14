"""API routes for Proxy Variable Hunter."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd, pickle, io, uuid
router = APIRouter()

@router.post("/hunt")
async def hunt_proxies(
    dataset_file: UploadFile = File(...),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
):
    from app.services.proxy_hunter import proxy_hunter_service
    audit_id = str(uuid.uuid4())[:8]
    try:
        df = pd.read_csv(io.BytesIO(await dataset_file.read()))
        protected = [c.strip() for c in protected_cols.split(",")]
        X = df[[c for c in df.columns if c != target_col]]
        result = await proxy_hunter_service.hunt_proxies(X, protected, audit_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
