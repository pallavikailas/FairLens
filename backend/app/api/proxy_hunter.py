"""API routes for Proxy Variable Hunter."""
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd, io, uuid

from app.services.builtin_datasets import load_builtin_dataset

router = APIRouter()


@router.post("/hunt")
async def hunt_proxies(
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    model_type: str = Form(default="sklearn"),
    test_suite: str = Form(default="auto"),
):
    from app.services.proxy_hunter import proxy_hunter_service

    audit_id = str(uuid.uuid4())[:8]
    try:
        # Always use hiring_bias for proxy analysis (richer tabular correlations)
        effective_suite = test_suite if test_suite != "auto" else "hiring_bias"
        dataset_csv, _auto_protected, _auto_target, _ = load_builtin_dataset(model_type, effective_suite)
        df = pd.read_csv(io.StringIO(dataset_csv))

        is_auto = protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(",")
        if is_auto:
            protected = _auto_protected
            tgt = _auto_target
        else:
            protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]
            tgt = target_col if target_col and target_col != "auto" else _auto_target

        if not tgt or tgt not in df.columns:
            for col in df.columns:
                if df[col].nunique() == 2:
                    tgt = col
                    break
            else:
                tgt = df.columns[-1]

        X = df[[c for c in df.columns if c != tgt]]
        y = pd.to_numeric(df[tgt], errors="coerce").fillna(0) if tgt in df.columns else None

        result = await proxy_hunter_service.run_hunt(X, y, protected, audit_id)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
