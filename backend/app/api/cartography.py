"""
Cartography API — auto-detects protected columns and target from dataset.
Supports: file upload, URL, HuggingFace datasets, Kaggle datasets.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uuid, io, httpx, pandas as pd

from app.services.cartography import cartography_service
from app.services.auto_detect import auto_detect_columns

router = APIRouter()


async def _load_dataset(
    dataset_file: Optional[UploadFile],
    dataset_source: str,
    dataset_url: str,
) -> str:
    """Load dataset CSV from any source and return as string."""

    if dataset_source == 'upload' or (dataset_file and dataset_file.filename):
        if not dataset_file:
            raise HTTPException(400, "dataset_file required for upload source")
        raw = await dataset_file.read()
        return raw.decode("utf-8", errors="replace")

    elif dataset_source == 'url' and dataset_url:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(dataset_url)
            resp.raise_for_status()
            return resp.text

    elif dataset_source == 'huggingface' and dataset_url:
        # Load from HuggingFace datasets API
        # dataset_url = "owner/dataset-name" or "dataset-name"
        name = dataset_url.strip()
        api_url = f"https://datasets-server.huggingface.co/first-rows?dataset={name}&config=default&split=train"
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(api_url)
            if resp.status_code != 200:
                # Fallback: try parquet viewer
                api_url = f"https://datasets-server.huggingface.co/rows?dataset={name}&config=default&split=train&offset=0&length=200"
                resp = await client.get(api_url)
            data = resp.json()
            rows = data.get("rows", [])
            if not rows:
                raise HTTPException(400, f"Could not load HuggingFace dataset '{name}'. Check the dataset name.")
            records = [r["row"] for r in rows]
            df = pd.DataFrame(records)
            return df.to_csv(index=False)

    elif dataset_source == 'kaggle' and dataset_url:
        raise HTTPException(400,
            "Kaggle datasets require an API key. Please download the CSV from Kaggle and upload it directly, "
            "or use the URL option with a direct CSV link."
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
        # Load dataset from whatever source
        dataset_csv = await _load_dataset(dataset_file, dataset_source, dataset_url)

        # Auto-detect protected columns and target if not specified
        if protected_cols == "auto" or not protected_cols:
            detected = await auto_detect_columns(dataset_csv, audit_id)
            protected = detected["protected_cols"]
            target = detected["target_col"]
        else:
            protected = [c.strip() for c in protected_cols.split(",") if c.strip()]
            target = target_col

        result = await cartography_service.run_cartography(
            dataset_csv=dataset_csv,
            protected_cols=protected,
            target_col=target,
            audit_id=audit_id,
        )

        # Include detected cols in response so frontend can update
        result["detected_protected_cols"] = protected
        result["detected_target_col"] = target
        result["model_type"] = model_type
        result["dataset_source"] = dataset_source

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cartography failed: {str(e)}")
