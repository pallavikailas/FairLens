"""Tests for FairLens Universal Model Adapter."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.services.model_adapter import FairLensAdapter, SklearnAdapter, CallableAdapter


@pytest.fixture
def binary_data():
    np.random.seed(0)
    X = pd.DataFrame({"age": np.random.randint(20, 60, 200), "score": np.random.rand(200), "gender": np.random.choice(["M", "F"], 200)})
    y = (X["score"] + np.where(X["gender"] == "M", 0.2, 0)).gt(0.5).astype(int)
    return X, y.values


@pytest.fixture
def trained_rf(binary_data):
    from sklearn.preprocessing import LabelEncoder
    X, y = binary_data
    Xe = X.copy()
    Xe["gender"] = LabelEncoder().fit_transform(Xe["gender"])
    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    clf.fit(Xe, y)
    return clf, Xe


def test_sklearn_adapter_predict(trained_rf):
    model, X = trained_rf
    adapter = FairLensAdapter.from_sklearn(model)
    preds = adapter.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_sklearn_adapter_predict_proba(trained_rf):
    model, X = trained_rf
    adapter = FairLensAdapter.from_sklearn(model)
    proba = adapter.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_sklearn_adapter_shap(trained_rf):
    model, X = trained_rf
    adapter = FairLensAdapter.from_sklearn(model)
    explainer = adapter.get_shap_explainer(X)
    assert explainer is not None


def test_callable_adapter():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.8, 0.4]})

    def my_predict(df):
        return (df["b"] > 0.5).astype(int).values

    def my_proba(df):
        p = df["b"].values
        return np.column_stack([1 - p, p])

    adapter = FairLensAdapter.from_callable(my_predict, my_proba, "MyModel")
    assert adapter.predict(X).tolist() == [0, 1, 0]
    proba = adapter.predict_proba(X)
    assert proba.shape == (3, 2)
    assert adapter.get_model_type() == "MyModel"


def test_auto_detect_sklearn(trained_rf):
    model, X = trained_rf
    adapter = FairLensAdapter.auto_detect(model)
    assert isinstance(adapter, SklearnAdapter)
    preds = adapter.predict(X)
    assert len(preds) == len(X)


def test_pickle_roundtrip(trained_rf, tmp_path):
    import pickle
    model, X = trained_rf
    pkl_path = tmp_path / "model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    adapter = FairLensAdapter.from_pickle(str(pkl_path))
    preds = adapter.predict(X)
    assert len(preds) == len(X)
