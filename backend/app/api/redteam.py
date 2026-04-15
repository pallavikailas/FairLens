"""API routes for Fairness Red-Team Agent — streams progress via SSE."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
import pandas as pd, numpy as np, pickle, io, uuid, json, asyncio, httpx

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
    except Exception as e:
        raise HTTPException(400, f"Failed to load model file: {e}")

    dataset_csv = await _load_dataset(dataset_file, dataset_source, dataset_url)
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

    X = df[[c for c in df.columns if c != tgt]]
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
