"""API routes for Proxy Variable Hunter."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd, io, uuid, httpx

router = APIRouter()


async def _load_dataset(
    dataset_file: Optional[UploadFile],
    dataset_source: str,
    dataset_url: str,
) -> str:
    """Load dataset CSV from upload or remote URL."""
    if dataset_source == "upload" or not dataset_url:
        if not dataset_file:
            raise HTTPException(400, "dataset_file is required when dataset_source is 'upload'")
        raw = await dataset_file.read()
        return raw.decode("utf-8", errors="replace")
    elif dataset_source in ("url", "huggingface", "kaggle"):
        if not dataset_url:
            raise HTTPException(400, "dataset_url required for non-upload dataset sources")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(dataset_url)
            resp.raise_for_status()
            return resp.text
    raise HTTPException(400, "No valid dataset source provided")


@router.post("/hunt")
async def hunt_proxies(
    dataset_file: Optional[UploadFile] = File(default=None),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    dataset_source: str = Form(default="upload"),
    dataset_url: str = Form(default=""),
):
    from app.services.proxy_hunter import proxy_hunter_service
    from app.services.auto_detect import auto_detect_columns

    audit_id = str(uuid.uuid4())[:8]
    try:
        dataset_csv = await _load_dataset(dataset_file, dataset_source, dataset_url)
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

        X = df[[c for c in df.columns if c != tgt]]
        result = proxy_hunter_service.hunt_proxies(X, protected, audit_id)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
