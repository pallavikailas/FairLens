"""API routes for Fairness Red-Team Agent — streams progress via SSE."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List
import pandas as pd, numpy as np, pickle, io, uuid, json, asyncio, logging

from app.services.dataset_loader import load_dataset_csv

logger = logging.getLogger(__name__)
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
    model_type: str = Form(default="sklearn"),
    api_endpoint: str = Form(default=""),
    llm_api_key: str = Form(default=""),
    hf_token: str = Form(default=""),
):
    """
    Runs the red-team agent and streams progress as Server-Sent Events.
    Accepts pkl models, HuggingFace, OpenAI, Gemini, and REST API endpoints.
    """
    from app.services.redteam import redteam_agent
    from app.services.auto_detect import auto_detect_columns
    from app.services.model_adapter import FairLensAdapter

    model = None

    # Load model based on type
    if model_file is not None:
        model_bytes = await model_file.read()
        if model_bytes:
            try:
                try:
                    raw = pickle.loads(model_bytes)
                except Exception:
                    import joblib
                    raw = joblib.load(io.BytesIO(model_bytes))
                model = FairLensAdapter.auto_detect(raw)
            except Exception as e:
                raise HTTPException(400, f"Failed to load model file: {e}")

    elif model_type == "huggingface" and api_endpoint:
        try:
            model = FairLensAdapter.from_huggingface(api_endpoint, task="text-classification", hf_token=hf_token)
        except Exception as e:
            raise HTTPException(400, f"Failed to load HuggingFace model '{api_endpoint}': {e}")

    elif model_type == "llm_hf" and api_endpoint:
        try:
            model = FairLensAdapter.from_generative_huggingface(api_endpoint, hf_token=hf_token)
        except Exception as e:
            raise HTTPException(400, f"Failed to configure HF generative model '{api_endpoint}': {e}")

    elif model_type == "openai" and api_endpoint:
        try:
            model = FairLensAdapter.from_openai(model_name=api_endpoint, api_key=llm_api_key)
        except Exception as e:
            raise HTTPException(400, f"Failed to configure OpenAI model '{api_endpoint}': {e}")

    elif model_type == "gemini_llm" and api_endpoint:
        try:
            model = FairLensAdapter.from_gemini(model_name=api_endpoint, api_key=llm_api_key)
        except Exception as e:
            raise HTTPException(400, f"Failed to configure Gemini model '{api_endpoint}': {e}")

    elif model_type == "api" and api_endpoint:
        try:
            model = FairLensAdapter.from_api(api_endpoint)
        except Exception as e:
            raise HTTPException(400, f"Failed to configure REST API adapter for '{api_endpoint}': {e}")

    if model is None:
        raise HTTPException(400, "No model provided. Upload a .pkl file or specify a model endpoint.")

    try:
        dataset_csv = await load_dataset_csv(dataset_file, dataset_source, dataset_url)
    except Exception as e:
        raise HTTPException(400, f"Failed to load dataset: {e}")

    try:
        df = pd.read_csv(io.StringIO(dataset_csv))
    except Exception as e:
        raise HTTPException(400, f"Failed to parse dataset CSV: {e}")

    # Resolve protected_cols and target_col — handle 'auto' sentinel
    is_auto = protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(",")
    if is_auto:
        try:
            detected = await auto_detect_columns(dataset_csv, str(uuid.uuid4())[:8])
            protected = detected["protected_cols"]
            tgt = detected["target_col"]
        except Exception as e:
            raise HTTPException(500, f"Auto-detect columns failed: {e}")
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

    # Try to resolve feature columns from the underlying model object
    raw_model = getattr(model, "model", None) or model  # SklearnAdapter.model = raw sklearn
    X = df[_resolve_feature_cols(raw_model, df, tgt)]
    y = df[tgt].values if tgt in df.columns else np.zeros(len(df), dtype=int)

    try:
        biases = json.loads(confirmed_biases)
    except Exception as e:
        raise HTTPException(400, f"confirmed_biases is not valid JSON: {e}")

    # audit_results is optional context — the agent nodes don't need the full output
    try:
        audit = json.loads(audit_results) if audit_results else {}
    except Exception:
        audit = {}

    audit_id = str(uuid.uuid4())[:8]

    def _safe_json(obj):
        """Convert an SSE event dict to a JSON string, skipping non-serializable values."""
        def default(o):
            import pandas as pd
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, pd.DataFrame):
                return f"<DataFrame {o.shape}>"
            return str(o)
        return json.dumps(obj, default=default)

    async def event_stream():
        try:
            async for event in redteam_agent.run(model, X, y, audit, biases, audit_id):
                yield f"data: {_safe_json(event)}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            logger.error(f"[{audit_id}] Red-team agent error: {err_msg}")
            yield f"data: {json.dumps({'node': 'error', 'status': 'error', 'log': [f'Agent error: {e}']})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
