"""
FairLens Universal Model Adapter
=================================
The plugin layer that makes FairLens model-agnostic.

Any model — sklearn, PyTorch, TensorFlow, HuggingFace, XGBoost, LightGBM,
a REST API, a LLM, or your own custom class — can be wrapped in a FairLensAdapter
so FairLens can audit it.

Usage:
    from app.services.model_adapter import FairLensAdapter

    # scikit-learn
    adapter = FairLensAdapter.from_sklearn(my_rf_model)

    # PyTorch
    adapter = FairLensAdapter.from_pytorch(my_net, input_size=10)

    # HuggingFace
    adapter = FairLensAdapter.from_huggingface("bert-base-uncased", task="text-classification")

    # REST API (any model behind an endpoint)
    adapter = FairLensAdapter.from_api("https://my-model-api.com/predict")

    # Custom callable
    adapter = FairLensAdapter.from_callable(my_predict_fn, my_proba_fn)

All adapters expose the same interface:
    adapter.predict(X: pd.DataFrame) -> np.ndarray
    adapter.predict_proba(X: pd.DataFrame) -> np.ndarray   # shape (n, 2)
    adapter.get_model_type() -> str
    adapter.supports_shap() -> bool
    adapter.get_shap_explainer(X_background) -> shap.Explainer
"""

from __future__ import annotations

import abc
import json
import logging
from typing import Any, Callable, Optional, List

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseModelAdapter(abc.ABC):
    """
    Abstract interface every FairLens adapter must implement.
    Implement this class to plug ANY model into FairLens.
    """

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions, shape (n,)."""

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates, shape (n, 2) for binary classification."""

    def get_model_type(self) -> str:
        return self.__class__.__name__

    def supports_shap(self) -> bool:
        """Override to False for models that can't use TreeExplainer."""
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        """
        Return the best SHAP explainer for this model type.
        Override for custom behaviour.
        """
        try:
            return shap.TreeExplainer(self._raw_model())
        except Exception:
            bg = shap.sample(X_background, min(100, len(X_background)))
            return shap.KernelExplainer(self.predict_proba, bg)

    def _raw_model(self) -> Any:
        """Return the underlying model object, if available."""
        raise NotImplementedError


# ── sklearn adapter ───────────────────────────────────────────────────────────

class SklearnAdapter(BaseModelAdapter):
    """
    Wraps any scikit-learn compatible estimator.
    Supports: RandomForest, XGBoost, LightGBM, LogisticRegression,
              SVM, GradientBoosting, CatBoost, and any Pipeline.
    """

    TREE_MODELS = (
        "RandomForest", "GradientBoosting", "XGB", "LGBM",
        "CatBoost", "DecisionTree", "ExtraTrees", "HistGradientBoosting"
    )

    def __init__(self, model: Any):
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_prep = self._prepare(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_prep)
            if proba.ndim == 1:
                return np.column_stack([1 - proba, proba])
            return proba
        # Decision function fallback (SVM etc.)
        scores = self.model.decision_function(X_prep)
        from scipy.special import expit
        pos = expit(scores)
        return np.column_stack([1 - pos, pos])

    def get_model_type(self) -> str:
        return type(self.model).__name__

    def supports_shap(self) -> bool:
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        model_name = type(self.model).__name__
        if any(t in model_name for t in self.TREE_MODELS):
            return shap.TreeExplainer(self.model)
        bg = shap.sample(self._prepare(X_background), min(100, len(X_background)))
        return shap.KernelExplainer(self.predict_proba, bg)

    def _raw_model(self) -> Any:
        return self.model

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Auto-encode categoricals for models that need numeric input."""
        from sklearn.preprocessing import LabelEncoder
        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=["object", "category"]).columns:
            try:
                X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
            except Exception:
                X_enc[col] = 0
        return X_enc.fillna(0)


# ── PyTorch adapter ───────────────────────────────────────────────────────────

class PyTorchAdapter(BaseModelAdapter):
    """
    Wraps a PyTorch nn.Module for binary or multi-class classification.
    The model must accept a float tensor of shape (n, input_size).
    """

    def __init__(self, model: Any, input_size: int, device: str = "cpu", threshold: float = 0.5):
        self.model = model
        self.input_size = input_size
        self.device = device
        self.threshold = threshold
        self.model.eval()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        tensor = torch.tensor(
            X.select_dtypes(include=[np.number]).fillna(0).values,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            if out.shape[-1] == 1:
                pos = torch.sigmoid(out).squeeze(-1).cpu().numpy()
                return np.column_stack([1 - pos, pos])
            proba = F.softmax(out, dim=-1).cpu().numpy()
            if proba.shape[1] == 2:
                return proba
            # Multi-class: return [1-max_prob, max_prob] as binary proxy
            max_p = proba.max(axis=1)
            return np.column_stack([1 - max_p, max_p])

    def supports_shap(self) -> bool:
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = X_background.select_dtypes(include=[np.number]).fillna(0).values[:100]
        return shap.GradientExplainer(self.model, bg)

    def get_model_type(self) -> str:
        return f"PyTorch:{type(self.model).__name__}"


# ── TensorFlow / Keras adapter ────────────────────────────────────────────────

class TensorFlowAdapter(BaseModelAdapter):
    """
    Wraps a tf.keras.Model or any TensorFlow SavedModel.
    """

    def __init__(self, model: Any, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import tensorflow as tf
        arr = X.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
        out = self.model.predict(arr, verbose=0)
        if out.ndim == 1 or out.shape[-1] == 1:
            pos = out.flatten()
            return np.column_stack([1 - pos, pos])
        return out[:, :2]

    def supports_shap(self) -> bool:
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = X_background.select_dtypes(include=[np.number]).fillna(0).values[:100]
        return shap.GradientExplainer(self.model, bg)

    def get_model_type(self) -> str:
        return f"TensorFlow:{type(self.model).__name__}"


# ── HuggingFace adapter ───────────────────────────────────────────────────────

class HuggingFaceAdapter(BaseModelAdapter):
    """
    Wraps a HuggingFace pipeline or AutoModel for text classification.
    Input DataFrame must contain a 'text' column.
    """

    def __init__(self, pipeline_or_model_name: Any, task: str = "text-classification"):
        from transformers import pipeline as hf_pipeline
        if isinstance(pipeline_or_model_name, str):
            self.pipeline = hf_pipeline(task, model=pipeline_or_model_name)
        else:
            self.pipeline = pipeline_or_model_name
        self._labels = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if "text" not in X.columns:
            raise ValueError("HuggingFaceAdapter requires a 'text' column in the DataFrame")
        texts = X["text"].fillna("").tolist()
        results = self.pipeline(texts, truncation=True, max_length=512)
        probas = []
        for r in results:
            if isinstance(r, list):
                r = r[0]
            score = r.get("score", 0.5)
            label = r.get("label", "").upper()
            # Treat POSITIVE / LABEL_1 / 1 as positive class
            if any(pos in label for pos in ["POS", "LABEL_1", "1"]):
                probas.append([1 - score, score])
            else:
                probas.append([score, 1 - score])
        return np.array(probas)

    def supports_shap(self) -> bool:
        return False  # Use LIME or KernelExplainer fallback

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        return shap.Explainer(
            lambda texts: self.predict_proba(pd.DataFrame({"text": texts})),
            X_background["text"].tolist() if "text" in X_background.columns else [""],
            output_names=["negative", "positive"]
        )

    def get_model_type(self) -> str:
        return "HuggingFace"


# ── REST API adapter ──────────────────────────────────────────────────────────

class RESTAPIAdapter(BaseModelAdapter):
    """
    Wraps any model served behind a REST endpoint.
    Sends rows as JSON and parses the response.

    Expected API contract:
        POST /predict
        Body: {"instances": [[f1, f2, ...], ...]}
        Response: {"predictions": [0, 1, ...], "probabilities": [[0.3,0.7], ...]}

    Override _format_request / _parse_response to match your API's schema.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[dict] = None,
        timeout: int = 30,
        request_format: str = "instances",   # or "data", "inputs" etc.
        auth_token: Optional[str] = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.headers = headers or {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        self.timeout = timeout
        self.request_format = request_format

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import requests
        payload = self._format_request(X)
        resp = requests.post(
            f"{self.endpoint}/predict",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_response(resp.json(), len(X))

    def supports_shap(self) -> bool:
        return False  # Use KernelExplainer

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = shap.sample(X_background, min(50, len(X_background)))
        return shap.KernelExplainer(self.predict_proba, bg)

    def _format_request(self, X: pd.DataFrame) -> dict:
        return {self.request_format: X.fillna(0).values.tolist()}

    def _parse_response(self, response: dict, n: int) -> np.ndarray:
        if "probabilities" in response:
            proba = np.array(response["probabilities"])
            if proba.ndim == 1:
                return np.column_stack([1 - proba, proba])
            return proba
        if "predictions" in response:
            preds = np.array(response["predictions"])
            return np.column_stack([1 - preds, preds.astype(float)])
        if "scores" in response:
            from scipy.special import expit
            scores = expit(np.array(response["scores"]))
            return np.column_stack([1 - scores, scores])
        raise ValueError(f"Unrecognised API response format: {list(response.keys())}")

    def get_model_type(self) -> str:
        return f"REST:{self.endpoint}"


# ── Callable adapter ──────────────────────────────────────────────────────────

class CallableAdapter(BaseModelAdapter):
    """
    Wraps any Python callable as a FairLens model.
    Useful for custom ensembles, business-rule systems, or any predict function.

    adapter = FairLensAdapter.from_callable(
        predict_fn=lambda X: my_model.predict(X),
        predict_proba_fn=lambda X: my_model.predict_proba(X),
    )
    """

    def __init__(
        self,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
        predict_proba_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        model_name: str = "CustomCallable",
    ):
        self._predict_fn = predict_fn
        self._predict_proba_fn = predict_proba_fn
        self._name = model_name

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(self._predict_fn(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._predict_proba_fn:
            proba = np.array(self._predict_proba_fn(X))
            if proba.ndim == 1:
                return np.column_stack([1 - proba, proba])
            return proba
        preds = self.predict(X).astype(float)
        return np.column_stack([1 - preds, preds])

    def supports_shap(self) -> bool:
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(self.predict_proba, bg)

    def get_model_type(self) -> str:
        return self._name


# ── Vertex AI / GCP Model adapter ─────────────────────────────────────────────

class VertexAIAdapter(BaseModelAdapter):
    """
    Wraps a deployed Vertex AI Endpoint.
    Calls the endpoint via google-cloud-aiplatform SDK.
    """

    def __init__(self, endpoint_id: str, project: str, location: str = "us-central1"):
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)
        self.endpoint = aiplatform.Endpoint(endpoint_id)
        self.project = project
        self.location = location

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        instances = X.fillna(0).to_dict(orient="records")
        response = self.endpoint.predict(instances=instances)
        preds = np.array(response.predictions)
        if preds.ndim == 1:
            return np.column_stack([1 - preds, preds])
        return preds

    def supports_shap(self) -> bool:
        return False

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = shap.sample(X_background, min(50, len(X_background)))
        return shap.KernelExplainer(self.predict_proba, bg)

    def get_model_type(self) -> str:
        return f"VertexAI:{self.endpoint.resource_name}"


# ── Main FairLensAdapter factory ──────────────────────────────────────────────

class FairLensAdapter:
    """
    Factory class and public API for the FairLens plugin system.

    Usage examples:

        # scikit-learn / XGBoost / LightGBM / CatBoost
        adapter = FairLensAdapter.from_sklearn(model)

        # PyTorch
        adapter = FairLensAdapter.from_pytorch(net, input_size=20)

        # TensorFlow/Keras
        adapter = FairLensAdapter.from_tensorflow(keras_model)

        # HuggingFace
        adapter = FairLensAdapter.from_huggingface("distilbert-base-uncased-finetuned-sst-2-english")

        # Any REST API
        adapter = FairLensAdapter.from_api("https://my-model.run.app", auth_token="abc123")

        # Any Python callable
        adapter = FairLensAdapter.from_callable(predict_fn, predict_proba_fn)

        # Vertex AI deployed endpoint
        adapter = FairLensAdapter.from_vertex_ai("1234567890", project="my-gcp-project")

        # Auto-detect from a .pkl file
        adapter = FairLensAdapter.from_pickle("model.pkl")
    """

    @staticmethod
    def from_sklearn(model: Any) -> SklearnAdapter:
        """Wrap any scikit-learn compatible model."""
        logger.info(f"Creating SklearnAdapter for {type(model).__name__}")
        return SklearnAdapter(model)

    @staticmethod
    def from_pytorch(model: Any, input_size: int, device: str = "cpu") -> PyTorchAdapter:
        """Wrap a PyTorch nn.Module."""
        logger.info(f"Creating PyTorchAdapter for {type(model).__name__}")
        return PyTorchAdapter(model, input_size=input_size, device=device)

    @staticmethod
    def from_tensorflow(model: Any) -> TensorFlowAdapter:
        """Wrap a Keras/TensorFlow model."""
        logger.info(f"Creating TensorFlowAdapter for {type(model).__name__}")
        return TensorFlowAdapter(model)

    @staticmethod
    def from_huggingface(
        model_name_or_pipeline: Any,
        task: str = "text-classification"
    ) -> HuggingFaceAdapter:
        """Wrap a HuggingFace model or pipeline."""
        logger.info(f"Creating HuggingFaceAdapter")
        return HuggingFaceAdapter(model_name_or_pipeline, task=task)

    @staticmethod
    def from_api(
        endpoint: str,
        headers: Optional[dict] = None,
        auth_token: Optional[str] = None,
        request_format: str = "instances",
    ) -> RESTAPIAdapter:
        """Wrap any REST API endpoint."""
        logger.info(f"Creating RESTAPIAdapter for {endpoint}")
        return RESTAPIAdapter(endpoint, headers=headers, auth_token=auth_token, request_format=request_format)

    @staticmethod
    def from_callable(
        predict_fn: Callable,
        predict_proba_fn: Optional[Callable] = None,
        model_name: str = "CustomCallable",
    ) -> CallableAdapter:
        """Wrap any Python predict function."""
        return CallableAdapter(predict_fn, predict_proba_fn, model_name)

    @staticmethod
    def from_vertex_ai(endpoint_id: str, project: str, location: str = "us-central1") -> VertexAIAdapter:
        """Wrap a Vertex AI deployed endpoint."""
        return VertexAIAdapter(endpoint_id, project=project, location=location)

    @staticmethod
    def from_pickle(path: str) -> SklearnAdapter:
        """Load a .pkl file and auto-wrap it."""
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {path}: {type(model).__name__}")
        return SklearnAdapter(model)

    @staticmethod
    def auto_detect(model: Any) -> BaseModelAdapter:
        """
        Auto-detect model type and return the appropriate adapter.
        Falls back to SklearnAdapter (which handles most cases).
        """
        class_name = type(model).__name__
        module = type(model).__module__ or ""

        if "torch" in module or "pytorch" in module.lower():
            raise ValueError("PyTorch models need input_size — use FairLensAdapter.from_pytorch(model, input_size=N)")

        if "keras" in module or "tensorflow" in module:
            return TensorFlowAdapter(model)

        if "transformers" in module:
            return HuggingFaceAdapter(model)

        # Default: treat as sklearn-compatible
        return SklearnAdapter(model)
