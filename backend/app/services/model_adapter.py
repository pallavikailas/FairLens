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


# ── Generative LLM adapter ────────────────────────────────────────────────────

class GenerativeLLMAdapter(BaseModelAdapter):
    """
    Wraps any generative LLM for bias auditing via decision prompts.

    Converts each tabular row into a decision scenario, asks the model to
    respond YES/NO, and returns predict_proba from that answer.
    Works with:
      - HuggingFace text-generation (Gemma, Llama, Mistral, etc.)
      - OpenAI  (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
      - Gemini  (gemini-1.5-flash, gemini-2.0-flash, etc.)

    Usage:
        adapter = GenerativeLLMAdapter(backend="openai",   model_name="gpt-4o",              api_key="sk-...")
        adapter = GenerativeLLMAdapter(backend="huggingface", model_name="google/gemma-3-1b-it", hf_token="hf_...")
        adapter = GenerativeLLMAdapter(backend="gemini",   model_name="gemini-2.0-flash",    api_key="AIza...")
    """

    _DEFAULT_PROMPT = (
        "You are an impartial decision-maker. Based only on the profile below, "
        "give a single-word decision: YES or NO.\n\n"
        "Profile:\n{profile}\n\n"
        "Decision (YES or NO):"
    )

    # HuggingFace Inference API endpoint — no local model download required
    _HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(
        self,
        backend: str,                        # "openai" | "huggingface" | "gemini"
        model_name: str,
        api_key: str = "",
        hf_token: str = "",
        prompt_template: str = "",
        max_new_tokens: int = 20,
        positive_threshold: float = 0.5,
    ):
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key
        self.hf_token = hf_token
        self.prompt_template = prompt_template or self._DEFAULT_PROMPT
        self.max_new_tokens = max_new_tokens
        self.positive_threshold = positive_threshold
        # HuggingFace uses the Inference API — no local pipeline/weights download

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.positive_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        for _, row in X.iterrows():
            p = self._query(row)
            probas.append([1.0 - p, p])
        return np.array(probas)

    def supports_shap(self) -> bool:
        return False

    def get_model_type(self) -> str:
        return f"GenerativeLLM:{self.backend}:{self.model_name}"

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _build_prompt(self, row: pd.Series) -> str:
        profile = "\n".join(f"  {k}: {v}" for k, v in row.items())
        return self.prompt_template.format(profile=profile)

    def _parse_response(self, text: str) -> float:
        """Return 0.9 for positive-class signals, 0.1 for negative, 0.5 for ambiguous."""
        t = text.lower().strip().split()[0] if text.strip() else ""
        if any(w in t for w in ("yes", "approv", "accept", "hire", "grant", "admit", "positive", "true", "1")):
            return 0.9
        if any(w in t for w in ("no", "reject", "deny", "declin", "negative", "false", "0")):
            return 0.1
        # Scan full text as fallback
        full = text.lower()
        pos = sum(full.count(w) for w in ("yes", "approv", "accept", "hire", "grant"))
        neg = sum(full.count(w) for w in ("no", "reject", "deny", "declin", "refused"))
        if pos > neg:
            return 0.75
        if neg > pos:
            return 0.25
        return 0.5

    # ── Backend query methods ─────────────────────────────────────────────────

    def _query(self, row: pd.Series) -> float:
        prompt = self._build_prompt(row)
        try:
            if self.backend == "openai":
                return self._query_openai(prompt)
            if self.backend == "huggingface":
                return self._query_huggingface(prompt)
            if self.backend == "gemini":
                return self._query_gemini(prompt)
        except Exception as exc:
            logger.warning(f"[GenerativeLLMAdapter] query failed: {exc}")
        return 0.5

    def _query_openai(self, prompt: str) -> float:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0,
        )
        return self._parse_response(resp.choices[0].message.content or "")

    def _query_huggingface(self, prompt: str) -> float:
        """Call HuggingFace Inference API — no local model download."""
        import requests
        url = self._HF_INFERENCE_URL.format(model=self.model_name)
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "return_full_text": False,
                "temperature": 0.01,
            },
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code == 503:
            raise RuntimeError(
                f"HuggingFace model '{self.model_name}' is loading on the Inference API — "
                "wait ~20s and retry, or use a smaller/always-on model."
            )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            generated = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            generated = data.get("generated_text", "")
        else:
            generated = str(data)
        return self._parse_response(generated)

    def _query_gemini(self, prompt: str) -> float:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        resp = model.generate_content(prompt)
        return self._parse_response(resp.text or "")


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
    def from_openai(
        model_name: str = "gpt-4o",
        api_key: str = "",
        prompt_template: str = "",
    ) -> "GenerativeLLMAdapter":
        """Wrap OpenAI ChatGPT / GPT-4 for decision-prompt bias auditing."""
        logger.info(f"Creating GenerativeLLMAdapter (OpenAI:{model_name})")
        return GenerativeLLMAdapter(backend="openai", model_name=model_name, api_key=api_key, prompt_template=prompt_template)

    @staticmethod
    def from_generative_huggingface(
        model_name: str,
        hf_token: str = "",
        prompt_template: str = "",
    ) -> "GenerativeLLMAdapter":
        """Wrap a HuggingFace generative model (Gemma, Llama, Mistral, etc.)."""
        logger.info(f"Creating GenerativeLLMAdapter (HuggingFace:{model_name})")
        return GenerativeLLMAdapter(backend="huggingface", model_name=model_name, hf_token=hf_token, prompt_template=prompt_template)

    @staticmethod
    def from_gemini(
        model_name: str = "gemini-2.0-flash",
        api_key: str = "",
        prompt_template: str = "",
    ) -> "GenerativeLLMAdapter":
        """Wrap a Gemini model for decision-prompt bias auditing."""
        logger.info(f"Creating GenerativeLLMAdapter (Gemini:{model_name})")
        return GenerativeLLMAdapter(backend="gemini", model_name=model_name, api_key=api_key, prompt_template=prompt_template)

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
