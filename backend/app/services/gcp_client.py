"""GCP client singletons — BigQuery, Cloud Storage."""
from google.cloud import bigquery, storage
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class BigQueryClient:
    def __init__(self):
        try:
            self._client = bigquery.Client(project=settings.GOOGLE_CLOUD_PROJECT)
        except Exception as e:
            logger.warning(f"BigQuery client init failed (non-fatal in local dev): {e}")
            self._client = None

    async def insert_rows(self, table_id: str, rows: list):
        if not self._client:
            return
        table_ref = f"{settings.GOOGLE_CLOUD_PROJECT}.{settings.BIGQUERY_DATASET}.{table_id}"
        errors = self._client.insert_rows_json(table_ref, rows)
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")


class StorageClient:
    def __init__(self):
        try:
            self._client = storage.Client(project=settings.GOOGLE_CLOUD_PROJECT)
        except Exception as e:
            logger.warning(f"Storage client init failed (non-fatal in local dev): {e}")
            self._client = None

    async def upload(self, data: bytes, blob_name: str, content_type: str = "application/json"):
        if not self._client:
            return None
        bucket = self._client.bucket(settings.GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(data, content_type=content_type)
        return f"gs://{settings.GCS_BUCKET_NAME}/{blob_name}"


bigquery_client = BigQueryClient()
storage_client = StorageClient()
