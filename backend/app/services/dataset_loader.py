"""
Shared dataset loading utility.

Supports: file upload, direct URL, HuggingFace datasets (with multi-fallback + 429 retry).
Returns CSV string in all cases so callers can pd.read_csv(io.StringIO(csv)).
"""
import asyncio
import io
import logging

import httpx
import pandas as pd
from fastapi import HTTPException, UploadFile
from typing import Optional

logger = logging.getLogger(__name__)

_HF_SERVER = "https://datasets-server.huggingface.co"
_HF_HUB = "https://huggingface.co"
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB


async def _get_with_retry(client: httpx.AsyncClient, url: str, max_retries: int = 3, **kwargs) -> httpx.Response:
    """GET with exponential back-off on 429 responses."""
    for attempt in range(max_retries):
        resp = await client.get(url, **kwargs)
        if resp.status_code != 429:
            return resp
        wait = 2 ** attempt  # 1s, 2s, 4s
        logger.info(f"[HF] 429 rate-limited on {url} — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
        await asyncio.sleep(wait)
    return resp  # return last response even if still 429


async def load_dataset_csv(
    dataset_file: Optional[UploadFile],
    dataset_source: str,
    dataset_url: str,
) -> str:
    """Load a dataset from any supported source and return it as a CSV string."""
    if dataset_source == "upload" or not dataset_url:
        if not dataset_file:
            raise HTTPException(400, "dataset_file is required when dataset_source is 'upload'")
        raw = await dataset_file.read()
        return raw.decode("utf-8", errors="replace")

    if dataset_source == "url":
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(dataset_url)
            resp.raise_for_status()
            return resp.text

    if dataset_source == "huggingface":
        return await _load_huggingface(dataset_url.strip())

    if dataset_source == "kaggle":
        raise HTTPException(
            400,
            "Kaggle requires authentication. Download the CSV from Kaggle and upload it directly."
        )

    raise HTTPException(400, f"Unknown dataset_source: '{dataset_source}'")


async def _load_huggingface(name: str) -> str:
    """
    Load a HuggingFace dataset by trying three strategies in order:
      1. datasets-server /rows  (instant, works for indexed datasets)
      2. Hub file listing → download first small CSV / JSON / JSONL file
      3. datasets-server /parquet → download first parquet shard < 100 MB
    All HTTP requests retry on 429 with exponential back-off.
    """
    async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:

        # --- Strategy 1: datasets-server /rows ---
        config, split = "default", "train"
        try:
            info = (await _get_with_retry(client, f"{_HF_SERVER}/info?dataset={name}")).json()
            configs = list(info.get("dataset_info", {}).keys())
            if configs:
                config = configs[0]
            splits = list(info.get("dataset_info", {}).get(config, {}).get("splits", {}).keys())
            if splits:
                split = splits[0]
        except Exception:
            pass

        rows_resp = await _get_with_retry(
            client,
            f"{_HF_SERVER}/rows?dataset={name}&config={config}&split={split}&offset=0&length=500"
        )
        if rows_resp.status_code == 200:
            rows = rows_resp.json().get("rows", [])
            if rows:
                logger.info(f"[HF] Loaded {len(rows)} rows via datasets-server for {name}")
                return pd.DataFrame([r["row"] for r in rows]).to_csv(index=False)

        # --- Strategy 2: Hub file listing → CSV/JSON ---
        try:
            tree_resp = await _get_with_retry(client, f"{_HF_HUB}/api/datasets/{name}/tree/main")
            if tree_resp.status_code == 200:
                tree = tree_resp.json()
                candidates = [
                    f for f in tree if isinstance(f, dict)
                    and f.get("path", "").lower().endswith((".csv", ".tsv", ".jsonl", ".json"))
                    and f.get("size", _MAX_FILE_BYTES + 1) < _MAX_FILE_BYTES
                ]
                candidates.sort(key=lambda f: f.get("size", 0))
                for f in candidates[:3]:
                    path = f["path"]
                    dl = await _get_with_retry(client, f"{_HF_HUB}/datasets/{name}/resolve/main/{path}")
                    if dl.status_code != 200:
                        continue
                    try:
                        if path.lower().endswith(".tsv"):
                            df = pd.read_csv(io.StringIO(dl.text), sep="\t", nrows=500)
                        elif path.lower().endswith(".jsonl"):
                            df = pd.read_json(io.StringIO(dl.text), lines=True, nrows=500)
                        elif path.lower().endswith(".json"):
                            df = pd.read_json(io.StringIO(dl.text), nrows=500)
                        else:
                            df = pd.read_csv(io.StringIO(dl.text), nrows=500)
                        logger.info(f"[HF] Loaded {len(df)} rows from {path} for {name}")
                        return df.to_csv(index=False)
                    except Exception as e:
                        logger.debug(f"[HF] Failed to parse {path}: {e}")
                        continue
        except Exception as e:
            logger.debug(f"[HF] Hub tree listing failed: {e}")

        # --- Strategy 3: parquet shards (with per-shard 429 retry) ---
        try:
            parquet_resp = await _get_with_retry(client, f"{_HF_SERVER}/parquet?dataset={name}")
            if parquet_resp.status_code == 200:
                pfiles = sorted(
                    parquet_resp.json().get("parquet_files", []),
                    key=lambda x: x.get("size", _MAX_FILE_BYTES + 1),
                )
                for pf in pfiles[:6]:  # try more shards to survive 429s
                    if pf.get("size", _MAX_FILE_BYTES + 1) > _MAX_FILE_BYTES:
                        continue
                    dl = await _get_with_retry(client, pf["url"])
                    if dl.status_code != 200:
                        logger.debug(f"[HF] Parquet shard {pf.get('url','')[:60]} returned {dl.status_code}")
                        continue
                    try:
                        import pyarrow.parquet as pq
                        table = pq.read_table(io.BytesIO(dl.content))
                        df = table.slice(0, min(500, table.num_rows)).to_pandas()
                        logger.info(f"[HF] Loaded {len(df)} rows from parquet shard for {name}")
                        return df.to_csv(index=False)
                    except Exception as e:
                        logger.debug(f"[HF] Failed to read parquet shard: {e}")
                        continue
        except Exception as e:
            logger.debug(f"[HF] Parquet fallback failed: {e}")

    raise HTTPException(
        400,
        f"Could not load HuggingFace dataset '{name}'. "
        f"The dataset may be gated, too large, or not indexed. "
        f"Download the CSV from huggingface.co/datasets/{name} and upload it directly."
    )
