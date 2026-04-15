"""
Cartography API — cloud-native version.
Accepts dataset from upload or online sources, auto-detects columns,
calls Gemini-powered BiasCartographyService (no SHAP/UMAP).
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
import uuid, io, httpx, pickle, logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.services.cartography import cartography_service
from app.services.auto_detect import auto_detect_columns

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_feature_cols(model, df: pd.DataFrame, fallback_target: str) -> List[str]:
    """
    Return the feature columns the model actually expects.
    Prefers the model's own stored feature names (set during fit) over guessing
    from target_col, which avoids mismatch errors when auto-detection picks
    the wrong target.
    """
    # sklearn >= 1.0 and XGBoost sklearn API store feature_names_in_
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        if names and all(n in df.columns for n in names):
            return names
    # XGBoost native booster stores feature_names
    if hasattr(model, "feature_names") and model.feature_names:
        names = model.feature_names
        if all(n in df.columns for n in names):
            return list(names)
    # LightGBM exposes feature_name() method
    if hasattr(model, "feature_name_"):
        try:
            names = model.feature_name_()
            if names and all(n in df.columns for n in names):
                return list(names)
        except Exception:
            pass
    # Fall back to excluding the detected target column
    return [c for c in df.columns if c != fallback_target]


async def _load_csv_string(
    dataset_file: Optional[UploadFile],
    dataset_source: str,
    dataset_url: str,
) -> str:
    """Load dataset from any source, return as CSV string."""

    if dataset_source == "upload" or not dataset_url:
        if not dataset_file:
            raise HTTPException(400, "dataset_file required for upload source")
        raw = await dataset_file.read()
        return raw.decode("utf-8", errors="replace")

    elif dataset_source == "url":
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(dataset_url)
            resp.raise_for_status()
            return resp.text

    elif dataset_source == "huggingface":
        name = dataset_url.strip()
        async with httpx.AsyncClient(timeout=30) as client:
            # Auto-discover config and split
            info_resp = await client.get(
                f"https://datasets-server.huggingface.co/info?dataset={name}"
            )
            config, split = "default", "train"
            if info_resp.status_code == 200:
                info = info_resp.json()
                configs = list(info.get("dataset_info", {}).keys())
                if configs:
                    config = configs[0]
                splits = list(
                    info.get("dataset_info", {})
                    .get(config, {})
                    .get("splits", {})
                    .keys()
                )
                if splits:
                    split = splits[0]

            rows_resp = await client.get(
                f"https://datasets-server.huggingface.co/rows"
                f"?dataset={name}&config={config}&split={split}&offset=0&length=500"
            )
            if rows_resp.status_code != 200:
                raise HTTPException(
                    400,
                    f"Could not load HuggingFace dataset '{name}'. "
                    f"Try downloading the CSV from huggingface.co/datasets/{name} and uploading it."
                )
            data = rows_resp.json()
            rows = data.get("rows", [])
            if not rows:
                raise HTTPException(400, f"HuggingFace dataset '{name}' returned no rows.")
            df = pd.DataFrame([r["row"] for r in rows])
            return df.to_csv(index=False)

    elif dataset_source == "kaggle":
        raise HTTPException(
            400,
            "Kaggle requires authentication. Please download the CSV from Kaggle and upload it directly."
        )

    raise HTTPException(400, "No valid dataset source provided")


@router.post("/analyze")
async def analyze_bias_cartography(
    dataset_file: Optional[UploadFile] = File(default=None),
    model_file: Optional[UploadFile] = File(default=None),
    protected_cols: str = Form(default="auto"),
    target_col: str = Form(default="auto"),
    model_type: str = Form(default="sklearn"),
    api_endpoint: str = Form(default=""),
    vertex_endpoint_id: str = Form(default=""),
    gcp_project: str = Form(default=""),
    dataset_source: str = Form(default="upload"),
    dataset_url: str = Form(default=""),
):
    audit_id = str(uuid.uuid4())[:8]
    try:
        # 1. Load dataset as CSV string
        dataset_csv = await _load_csv_string(dataset_file, dataset_source, dataset_url)

        # 2. Auto-detect or use provided columns
        is_auto = protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(",")
        if is_auto:
            detected = await auto_detect_columns(dataset_csv, audit_id)
            protected = detected["protected_cols"]
            target = detected["target_col"]
        else:
            protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]
            target = target_col

        # Fallback if detection found nothing
        if not protected or not target:
            df_check = pd.read_csv(io.StringIO(dataset_csv))
            if not protected:
                protected = []
            if not target or target not in df_check.columns:
                for col in df_check.columns:
                    if df_check[col].nunique() == 2:
                        target = col
                        break
                else:
                    target = df_check.columns[-1]

        # 3. Load model and generate predictions if provided
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
                    feature_cols = _resolve_feature_cols(clf, df_pred, target)
                    # If model knows its own features, infer the true target as whatever column is left out
                    non_feature_cols = [c for c in df_pred.columns if c not in feature_cols]
                    if non_feature_cols and non_feature_cols[0] != target:
                        logger.info(f"[{audit_id}] Overriding auto-detected target '{target}' → '{non_feature_cols[0]}' (from model feature names)")
                        target = non_feature_cols[0]
                    X = df_pred[feature_cols]
                    # Try raw prediction first; fall back to label-encoded if model needs numerics
                    try:
                        preds = clf.predict(X)
                    except Exception:
                        X_enc = X.copy()
                        le = LabelEncoder()
                        for col in X_enc.select_dtypes(include=["object", "category"]).columns:
                            try:
                                X_enc[col] = le.fit_transform(X_enc[col].astype(str))
                            except Exception:
                                X_enc[col] = 0
                        preds = clf.predict(X_enc.fillna(0))
                    model_predictions = [int(p) for p in preds]
                    logger.info(f"[{audit_id}] Generated {len(model_predictions)} model predictions for bias analysis")
                except Exception as e:
                    logger.warning(f"[{audit_id}] Could not generate model predictions: {e} — falling back to dataset labels")

        # 4. Call cartography service with model predictions when available
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
        result["dataset_source"] = dataset_source
        result["analysis_source"] = "model_predictions" if model_predictions else "dataset_labels"

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cartography failed: {str(e)}")
