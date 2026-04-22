"""API routes for Counterfactual Constitution."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle, io, uuid, json, asyncio
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
    model_type: str = Form(default="sklearn"),
    api_endpoint: str = Form(default=""),
    llm_api_key: str = Form(default=""),
    hf_token: str = Form(default=""),
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
                    raw = pickle.loads(model_bytes)
                except Exception:
                    try:
                        import joblib
                        raw = joblib.load(io.BytesIO(model_bytes))
                    except Exception as e:
                        raise HTTPException(400, f"Failed to load model file: {e}")
                # Always wrap raw models so predict auto-encodes categoricals consistently
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.auto_detect(raw)

        elif model_type == "huggingface" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_huggingface_auto(api_endpoint, hf_token=hf_token)
                # HuggingFaceAdapter._to_text auto-serialises tabular rows when no 'text' column present
            except Exception as e:
                err = str(e)
                if "No module named 'transformers'" in err:
                    raise HTTPException(500, "transformers library not installed on the server. Contact the administrator.")
                if any(kw in err.lower() for kw in ["text-generation", "causal", "generative", "seq2seq", "not supported for the pipeline"]):
                    raise HTTPException(400, f"'{api_endpoint}' is a generative/text-generation model. FairLens requires a text-classification model. Try one of: unitary/toxic-bert, cardiffnlp/twitter-roberta-base-sentiment, valurank/distilroberta-base-offensive-language-identification")
                raise HTTPException(400, f"Failed to load HuggingFace model '{api_endpoint}': {e}")

        elif model_type == "api" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_api(api_endpoint)
            except Exception as e:
                raise HTTPException(400, f"Failed to connect to API model '{api_endpoint}': {e}")

        elif model_type == "llm_hf" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_generative_huggingface(api_endpoint, hf_token=hf_token)
            except Exception as e:
                raise HTTPException(400, f"Failed to load HuggingFace generative model '{api_endpoint}': {e}")

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

        feature_cols = _resolve_feature_cols(model, df, tgt) if model is not None else [c for c in df.columns if c != tgt]
        X = df[feature_cols]

        # Auto-train a reference model when none is provided so counterfactual simulation always runs
        if model is None:
            try:
                import warnings
                from sklearn.linear_model import LogisticRegression
                from app.services.model_adapter import FairLensAdapter

                y_train = pd.to_numeric(df[tgt], errors="coerce").fillna(0).values.astype(int)

                # Build fixed encoding maps so flipped attribute values encode consistently
                encoding_maps: dict = {}
                X_enc_train = X.copy()
                for col in X_enc_train.select_dtypes(include=["object", "category"]).columns:
                    try:
                        le_fit = LabelEncoder()
                        X_enc_train[col] = le_fit.fit_transform(X_enc_train[col].astype(str))
                        encoding_maps[col] = {str(v): i for i, v in enumerate(le_fit.classes_)}
                    except Exception:
                        X_enc_train[col] = 0
                X_enc_train = X_enc_train.fillna(0)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auto_clf = LogisticRegression(max_iter=500, random_state=42)
                    auto_clf.fit(X_enc_train, y_train)

                def _encode_df(X_in: pd.DataFrame) -> pd.DataFrame:
                    X_e = X_in.copy()
                    for col, mapping in encoding_maps.items():
                        if col in X_e.columns:
                            X_e[col] = X_e[col].astype(str).map(mapping).fillna(-1)
                    return X_e.fillna(0)

                model = FairLensAdapter.from_callable(
                    predict_fn=lambda X_in: auto_clf.predict(_encode_df(X_in)),
                    predict_proba_fn=lambda X_in: auto_clf.predict_proba(_encode_df(X_in)),
                    model_name="AutoReference_LogisticRegression",
                )
                import logging as _log
                _log.getLogger(__name__).info(
                    f"[{audit_id}] Auto-trained reference LogisticRegression for counterfactual simulation"
                )
            except Exception as _e:
                import logging as _log
                _log.getLogger(__name__).warning(
                    f"[{audit_id}] Auto-training failed, constitution will use dataset-only mode: {_e}"
                )

        # Generate predictions
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
                timeout=540,  # 9 min — well under Cloud Run's 1hr, avoids gateway timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                504,
                "Constitution timed out (>9 min). The model API is too slow for full counterfactual analysis. "
                "Try a faster model or reduce dataset size."
            )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
