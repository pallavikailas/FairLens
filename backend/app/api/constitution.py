"""API routes for Counterfactual Constitution."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
import numpy as np
import pickle, io, uuid, json, asyncio, logging

from app.services.constitution import constitution_service
from app.services.builtin_datasets import load_builtin_dataset
from app.api._utils import resolve_feature_cols

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate")
async def generate_constitution(
    model_file: Optional[UploadFile] = File(default=None),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    cartography_results: str = Form(..., description="JSON string from cartography stage"),
    model_type: str = Form(default="sklearn"),
    api_endpoint: str = Form(default=""),
    llm_api_key: str = Form(default=""),
    hf_token: str = Form(default=""),
    test_suite: str = Form(default="auto"),
):
    audit_id = str(uuid.uuid4())[:8]
    try:
        # Load the same built-in dataset used during cartography
        dataset_csv, _auto_protected, _auto_target, _ = load_builtin_dataset(model_type, test_suite)
        df = pd.read_csv(io.StringIO(dataset_csv))

        # Use caller-supplied columns (detected during cartography stage)
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

        # Load model
        model = None
        if model_file is not None:
            model_bytes = await model_file.read()
            if model_bytes:
                try:
                    raw = pickle.loads(model_bytes)
                except Exception:
                    try:
                        import joblib
                        raw = joblib.load(io.BytesIO(model_bytes))
                    except Exception as e:
                        raise HTTPException(400, f"Failed to load model file: {e}")
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.auto_detect(raw)

        elif model_type == "huggingface" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_huggingface_auto(api_endpoint, hf_token=hf_token)
            except Exception as e:
                err = str(e)
                if "No module named 'transformers'" in err:
                    raise HTTPException(500, "transformers library not installed.")
                if any(kw in err.lower() for kw in ["text-generation", "causal", "generative", "seq2seq"]):
                    raise HTTPException(400, f"'{api_endpoint}' is a generative model. FairLens requires a classifier.")
                raise HTTPException(400, f"Failed to load HuggingFace model '{api_endpoint}': {e}")

        elif model_type == "api" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_api(api_endpoint)
            except Exception as e:
                raise HTTPException(400, f"Failed to connect to API model '{api_endpoint}': {e}")

        elif model_type == "openai" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_openai(model_name=api_endpoint, api_key=llm_api_key)
            except Exception as e:
                raise HTTPException(400, f"Failed to initialise OpenAI model '{api_endpoint}': {e}")

        elif model_type == "gemini_llm" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_gemini(model_name=api_endpoint, api_key=llm_api_key)
            except Exception as e:
                raise HTTPException(400, f"Failed to initialise Gemini model '{api_endpoint}': {e}")

        if model is None:
            raise HTTPException(400, "No model provided. Upload a .pkl file or specify a model endpoint.")

        feature_cols = resolve_feature_cols(model, df, tgt)
        X = df[feature_cols]

        model_type_str = (model.get_model_type() if hasattr(model, "get_model_type") else "") or ""
        is_api_model = any(t in model_type_str for t in ("HuggingFace", "REST:", "GenerativeLLM"))
        pred_sample_size = 150 if is_api_model else len(X)

        X_pred = X.iloc[:pred_sample_size] if pred_sample_size < len(X) else X
        try:
            y_pred_sample = model.predict(X_pred)
            if pred_sample_size < len(X):
                import scipy.stats as _stats
                fill_val = int(_stats.mode(y_pred_sample, keepdims=True).mode[0])
                y_pred = np.full(len(X), fill_val, dtype=int)
                y_pred[:pred_sample_size] = y_pred_sample
            else:
                y_pred = y_pred_sample
        except Exception as e:
            raise HTTPException(400, f"Model prediction failed: {e}")

        carto = json.loads(cartography_results)

        try:
            result = await asyncio.wait_for(
                constitution_service.generate_constitution(
                    model=model,
                    X=X,
                    y_pred=y_pred,
                    protected_cols=protected,
                    feature_names=feature_cols,
                    cartography_results=carto,
                    audit_id=audit_id,
                ),
                timeout=540,
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Constitution timed out (>9 min).")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
