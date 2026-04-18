"""API routes for Fairness Red-Team Agent — streams progress via SSE."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
import pandas as pd, numpy as np, pickle, io, uuid, json, asyncio

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


@router.post("/run")
async def run_redteam(
    model_file: Optional[UploadFile] = File(default=None),
    dataset_file: Optional[UploadFile] = File(default=None),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    confirmed_biases: str = Form(..., description="JSON list of confirmed bias objects"),
    audit_results: str = Form(..., description="JSON of combined audit results"),
    dataset_source: str = Form(default="upload"),
    dataset_url: str = Form(default=""),
):
    """
    Runs the red-team agent and streams progress as Server-Sent Events.
    Requires a model file — returns 400 if not provided.
    """
    from app.services.redteam import redteam_agent
    from app.services.auto_detect import auto_detect_columns

    # Red-team requires a model to generate adversarial probes
    if model_file is None:
        raise HTTPException(400, "model_file is required for red-team analysis")

    model_bytes = await model_file.read()
    if not model_bytes:
        raise HTTPException(400, "model_file is empty")

    try:
        model = pickle.loads(model_bytes)
    except Exception:
        try:
            import joblib
            model = joblib.load(io.BytesIO(model_bytes))
        except Exception as e:
            raise HTTPException(400, f"Failed to load model file: {e}")

    dataset_csv = await load_dataset_csv(dataset_file, dataset_source, dataset_url)
    df = pd.read_csv(io.StringIO(dataset_csv))

    # Resolve protected_cols and target_col — handle 'auto' sentinel
    is_auto = protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(",")
    if is_auto:
        detected = await auto_detect_columns(dataset_csv, str(uuid.uuid4())[:8])
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

    X = df[_resolve_feature_cols(model, df, tgt)]
    y = df[tgt].values if tgt in df.columns else np.zeros(len(df), dtype=int)
    biases = json.loads(confirmed_biases)
    audit = json.loads(audit_results)
    audit_id = str(uuid.uuid4())[:8]

    async def event_stream():
        async for event in redteam_agent.run(model, X, y, audit, biases, audit_id):
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
