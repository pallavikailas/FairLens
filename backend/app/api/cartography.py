"""Cartography API — Gemini-powered bias mapping (no SHAP/UMAP)."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uuid, io, pickle, logging
import pandas as pd
import numpy as np

from app.services.cartography import cartography_service
from app.services.builtin_datasets import load_builtin_dataset
from app.services.compliance_mapper import check_compliance
from app.api._utils import resolve_feature_cols

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze")
async def analyze_bias_cartography(
    model_file: Optional[UploadFile] = File(default=None),
    model_type: str = Form(default="sklearn"),
    api_endpoint: str = Form(default=""),
    vertex_endpoint_id: str = Form(default=""),
    gcp_project: str = Form(default=""),
    llm_api_key: str = Form(default=""),
    hf_token: str = Form(default=""),
    test_suite: str = Form(default="auto"),
):
    audit_id = str(uuid.uuid4())[:8]
    try:
        # 1. Load the appropriate built-in dataset
        dataset_csv, protected, target, suite_name = load_builtin_dataset(model_type, test_suite)

        # 2. Load model and generate predictions
        model_predictions = None

        if model_file is not None:
            model_bytes = await model_file.read()
            if model_bytes:
                try:
                    try:
                        clf = pickle.loads(model_bytes)
                    except Exception:
                        import joblib
                        clf = joblib.load(io.BytesIO(model_bytes))
                    df_pred = pd.read_csv(io.StringIO(dataset_csv))
                    feature_cols = resolve_feature_cols(clf, df_pred, target)
                    non_feature_cols = [c for c in df_pred.columns if c not in feature_cols]
                    if non_feature_cols and non_feature_cols[0] != target:
                        logger.info(f"[{audit_id}] Overriding target '{target}' → '{non_feature_cols[0]}'")
                        target = non_feature_cols[0]
                    X = df_pred[feature_cols]
                    try:
                        preds = clf.predict(X)
                    except Exception:
                        from sklearn.preprocessing import LabelEncoder
                        X_enc = X.copy()
                        le = LabelEncoder()
                        for col in X_enc.select_dtypes(include=["object", "category"]).columns:
                            try:
                                X_enc[col] = le.fit_transform(X_enc[col].astype(str))
                            except Exception:
                                X_enc[col] = 0
                        preds = clf.predict(X_enc.fillna(0))
                    model_predictions = [int(p) for p in preds]
                    logger.info(f"[{audit_id}] Generated {len(model_predictions)} predictions from sklearn model")
                except Exception as e:
                    logger.warning(f"[{audit_id}] Could not generate model predictions: {e}")

        elif model_type == "huggingface" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                adapter = FairLensAdapter.from_huggingface_auto(api_endpoint, hf_token=hf_token)
                df_pred = pd.read_csv(io.StringIO(dataset_csv))
                if "text" not in df_pred.columns:
                    str_cols = df_pred.select_dtypes(include=["object"]).columns.tolist()
                    df_pred["text"] = df_pred[str_cols].fillna("").astype(str).agg(" ".join, axis=1) if str_cols else df_pred.astype(str).agg(" ".join, axis=1)
                hf_sample_size = min(300, len(df_pred))
                df_sample = df_pred.sample(hf_sample_size, random_state=42).reset_index(drop=True)
                preds_sample = adapter.predict(df_sample)
                model_predictions = [int(p) for p in preds_sample]
                dataset_csv = df_sample.to_csv(index=False)
                logger.info(f"[{audit_id}] HuggingFace '{api_endpoint}' predicted {hf_sample_size} rows")
            except Exception as e:
                logger.warning(f"[{audit_id}] HuggingFace predictions failed: {e}")

        elif model_type in ("openai", "gemini_llm") and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                adapter = (
                    FairLensAdapter.from_openai(model_name=api_endpoint, api_key=llm_api_key)
                    if model_type == "openai"
                    else FairLensAdapter.from_gemini(model_name=api_endpoint, api_key=llm_api_key)
                )
                df_pred = pd.read_csv(io.StringIO(dataset_csv))
                feature_cols_pred = [c for c in df_pred.columns if c != target]
                sample_size = min(100, len(df_pred))
                df_sample = df_pred.sample(sample_size, random_state=42).reset_index(drop=True)
                preds_sample = adapter.predict(df_sample[feature_cols_pred])
                model_predictions = [int(round(float(p))) for p in preds_sample]
                dataset_csv = df_sample.to_csv(index=False)
                logger.info(f"[{audit_id}] {model_type} '{api_endpoint}' predicted {sample_size} rows")
            except Exception as e:
                logger.warning(f"[{audit_id}] {model_type} predictions failed: {e}")

        elif model_type == "api" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                adapter = FairLensAdapter.from_api(api_endpoint)
                df_pred = pd.read_csv(io.StringIO(dataset_csv))
                feature_cols_pred = [c for c in df_pred.columns if c != target]
                preds = adapter.predict(df_pred[feature_cols_pred])
                model_predictions = [int(p) for p in preds]
                logger.info(f"[{audit_id}] REST API '{api_endpoint}' generated {len(model_predictions)} predictions")
            except Exception as e:
                logger.warning(f"[{audit_id}] REST API predictions failed: {e}")

        if model_predictions is None:
            raise HTTPException(
                400,
                "No model provided or model prediction failed. "
                "Upload a .pkl file or specify a model endpoint (HuggingFace, OpenAI, Gemini, REST API)."
            )

        # 3. Run cartography
        result = await cartography_service.run_cartography(
            dataset_csv=dataset_csv,
            protected_cols=protected,
            target_col=target,
            model_predictions=model_predictions,
            audit_id=audit_id,
        )

        result["detected_protected_cols"] = protected
        result["detected_target_col"] = target
        result["model_type"] = model_type
        result["test_suite"] = suite_name
        result["analysis_source"] = "model_predictions"
        result["compliance_tags"] = check_compliance(result.get("slice_metrics", []))

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cartography failed: {str(e)}")
