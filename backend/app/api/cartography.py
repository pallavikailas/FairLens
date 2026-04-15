"""
Cartography API — cloud-native version.
Accepts dataset from upload or online sources, auto-detects columns,
calls Gemini-powered BiasCartographyService (no SHAP/UMAP).
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uuid, io, httpx
import pandas as pd

from app.services.cartography import cartography_service
from app.services.auto_detect import auto_detect_columns

router = APIRouter()


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

        # 3. Call cloud-native cartography service (Gemini-powered, no SHAP/UMAP)
        result = await cartography_service.run_cartography(
            dataset_csv=dataset_csv,
            protected_cols=protected,
            target_col=target,
            audit_id=audit_id,
        )

        result["detected_protected_cols"] = protected
        result["detected_target_col"] = target
        result["model_type"] = model_type
        result["dataset_source"] = dataset_source

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cartography failed: {str(e)}")
