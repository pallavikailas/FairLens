"""Built-in bias test datasets bundled with FairLens.

Two curated suites:
  - text_probes   : 260 counterfactual text probes across race, gender, religion
                    (for text classifiers and LLM endpoints)
  - hiring_bias   : 500-row synthetic hiring dataset with injected gender/race bias
                    (for tabular sklearn / REST API models)

The correct suite is chosen automatically from model_type unless the caller
passes an explicit test_suite name.
"""
from __future__ import annotations
import os, pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Model types that consume free-text input
_TEXT_MODEL_TYPES = {"huggingface", "openai", "gemini_llm"}

SUITE_META = {
    "text_probes": {
        "file": "bias_text_probes.csv",
        "protected_cols": ["demographic_group", "bias_category"],
        "target_col": "true_label",
        "description": (
            "260 counterfactual text probes across 13 demographic groups "
            "(race, gender, religion). Identical templates with demographic "
            "terms substituted — an unbiased model should treat all groups equally."
        ),
    },
    "hiring_bias": {
        "file": "hiring_bias.csv",
        "protected_cols": ["gender", "race"],
        "target_col": "hired",
        "description": (
            "500-row synthetic hiring dataset with injected gender bias "
            "(+30 pp for males) and race bias (+20 pp for white candidates). "
            "Tests whether a classifier reproduces or amplifies these disparities."
        ),
    },
}


def resolve_suite(model_type: str, test_suite: str = "auto") -> str:
    """Return the canonical suite name for the given model_type."""
    if test_suite != "auto":
        if test_suite not in SUITE_META:
            raise ValueError(f"Unknown test_suite '{test_suite}'. Choose from: {list(SUITE_META)}")
        return test_suite
    return "text_probes" if model_type in _TEXT_MODEL_TYPES else "hiring_bias"


def load_builtin_dataset(
    model_type: str,
    test_suite: str = "auto",
) -> tuple[str, list[str], str, str]:
    """Load a built-in dataset and return (csv_str, protected_cols, target_col, suite_name)."""
    suite = resolve_suite(model_type, test_suite)
    meta = SUITE_META[suite]
    path = os.path.join(_DATA_DIR, meta["file"])
    df = pd.read_csv(path)
    return df.to_csv(index=False), meta["protected_cols"], meta["target_col"], suite
