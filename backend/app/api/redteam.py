"""API routes for Fairness Red-Team Agent — streams progress via SSE."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd, numpy as np, pickle, io, uuid, json, asyncio

router = APIRouter()

@router.post("/run")
async def run_redteam(
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    confirmed_biases: str = Form(..., description="JSON list of confirmed bias objects"),
    audit_results: str = Form(..., description="JSON of combined audit results"),
):
    """
    Runs the red-team agent and streams progress as Server-Sent Events.
    The frontend shows a live activity feed as the agent works.
    """
    from app.services.redteam import redteam_agent

    model = pickle.loads(await model_file.read())
    df = pd.read_csv(io.BytesIO(await dataset_file.read()))
    protected = [c.strip() for c in protected_cols.split(",")]
    X = df[[c for c in df.columns if c != target_col]]
    y = df[target_col].values
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
