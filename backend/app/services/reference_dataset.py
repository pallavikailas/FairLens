"""
Embedded Reference Dataset for Model Bias Probing
===================================================
A neutral synthetic dataset baked into the codebase — used to probe any model
for hidden demographic biases independently of the user-uploaded dataset.

Design principles:
  - Balanced demographic distributions (no historical skew)
  - Target labels derived from objective criteria only (credit_score + income)
  - Any disparity in model outputs across demographic groups = model bias, not data artefact
  - Covers the three most legally significant protected axes: gender, race, age_group

For sklearn/structured models the generation logic also creates a model-specific
probe by varying protected columns while keeping non-demographic features fixed.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

REFERENCE_PROTECTED_COLS = ["gender", "race", "age_group"]
REFERENCE_TARGET_COL = "outcome"

# Demographic values used across all reference probes
_GENDERS    = ["Male", "Female", "Non-binary"]
_RACES      = ["White", "Black", "Hispanic", "Asian", "Other"]
_AGE_GROUPS = ["18-30", "31-45", "46-60", "60+"]


def generate_reference_dataset(seed: int = 42) -> Tuple[pd.DataFrame, str]:
    """
    Generate a 300-row synthetic reference dataset for model probing.

    Returns
    -------
    (df, csv_string)
    """
    rng = np.random.default_rng(seed)
    n = 300

    genders    = rng.choice(_GENDERS,    n, p=[0.4, 0.4, 0.2])
    races      = rng.choice(_RACES,      n, p=[0.40, 0.20, 0.20, 0.15, 0.05])
    age_groups = rng.choice(_AGE_GROUPS, n, p=[0.25, 0.35, 0.25, 0.15])

    education  = rng.choice(["High School", "Bachelor", "Master", "PhD"],       n, p=[0.30, 0.40, 0.20, 0.10])
    employment = rng.choice(["Full-time", "Part-time", "Contract", "Self-employed"], n, p=[0.50, 0.20, 0.20, 0.10])

    years_exp    = rng.integers(0, 25, n).astype(int)
    credit_score = rng.integers(300, 850, n).astype(int)
    income_level = rng.integers(20_000, 150_000, n).astype(int)

    # Objective outcome: pure function of credit_score + income (zero demographic signal)
    outcome_score = (credit_score / 850) * 0.6 + (income_level / 150_000) * 0.4
    outcome = (outcome_score > 0.5).astype(int)

    df = pd.DataFrame({
        "gender":           genders,
        "race":             races,
        "age_group":        age_groups,
        "education_level":  education,
        "employment_type":  employment,
        "years_experience": years_exp,
        "credit_score":     credit_score,
        "income_level":     income_level,
        REFERENCE_TARGET_COL: outcome,
    })

    return df, df.to_csv(index=False)


def generate_model_specific_probe(
    model_feature_names: List[str],
    protected_cols: Optional[List[str]] = None,
    n: int = 240,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str, List[str], str, List[str]]:
    """
    Build a probe dataset that matches *model_feature_names*.

    Protected columns are identified by keyword matching; their values are varied
    systematically to expose demographic bias.  All other features receive generic
    numeric/categorical values so the model can always score the rows.

    When the model has no demographic features, standard gender/race/age_group columns
    are injected alongside the model features.  The returned ``model_feature_cols``
    list contains only the columns that should be passed to ``model.predict()``.

    Returns
    -------
    (df, csv_string, probe_protected_cols, probe_target_col, model_feature_cols)
    """
    rng = np.random.default_rng(seed)
    detected_protected: List[str] = []
    data: dict = {}

    _GENDER_KW    = {"gender", "sex"}
    _RACE_KW      = {"race", "ethnic", "ethnicity", "nationality"}
    _AGE_KW       = {"age"}
    _EDUCATION_KW = {"education", "edu", "degree", "qualification"}
    _EMPLOY_KW    = {"employ", "job", "occupation", "work"}
    _INCOME_KW    = {"income", "salary", "wage", "earning", "revenue"}
    _CREDIT_KW    = {"credit", "fico", "score"}
    _EXP_KW       = {"experience", "exp", "tenure", "seniority"}

    for feat in model_feature_names:
        fl = feat.lower().replace("_", " ").replace("-", " ")
        tokens = set(fl.split())

        if tokens & _GENDER_KW:
            data[feat] = rng.choice(_GENDERS, n, p=[0.4, 0.4, 0.2])
            detected_protected.append(feat)

        elif tokens & _RACE_KW:
            data[feat] = rng.choice(_RACES, n, p=[0.40, 0.20, 0.20, 0.15, 0.05])
            detected_protected.append(feat)

        elif tokens & _AGE_KW:
            if any(k in fl for k in ("group", "range", "bracket", "band", "category")):
                data[feat] = rng.choice(_AGE_GROUPS, n)
                detected_protected.append(feat)
            else:
                data[feat] = rng.integers(18, 80, n).astype(int)
                detected_protected.append(feat)

        elif tokens & _EDUCATION_KW:
            data[feat] = rng.choice(["High School", "Bachelor", "Master", "PhD"], n)

        elif tokens & _EMPLOY_KW:
            data[feat] = rng.choice(["Full-time", "Part-time", "Contract"], n)

        elif tokens & _INCOME_KW:
            data[feat] = rng.integers(20_000, 200_000, n).astype(int)

        elif tokens & _CREDIT_KW:
            data[feat] = rng.integers(300, 850, n).astype(int)

        elif tokens & _EXP_KW:
            data[feat] = rng.integers(0, 30, n).astype(int)

        else:
            # Generic fallback — numeric
            data[feat] = rng.integers(0, 100, n).astype(int)

    # Use supplied protected_cols override when the model doesn't have obvious keywords
    if protected_cols:
        detected_protected = [c for c in protected_cols if c in model_feature_names]

    # When model has no demographic features, inject standard demographics alongside
    # the model features so cartography can measure demographic disparity.
    # model_feature_cols tracks which columns to pass to model.predict().
    model_feature_cols = list(model_feature_names)
    if not detected_protected:
        data["gender"]    = rng.choice(_GENDERS,    n, p=[0.4, 0.4, 0.2])
        data["race"]      = rng.choice(_RACES,      n, p=[0.40, 0.20, 0.20, 0.15, 0.05])
        data["age_group"] = rng.choice(_AGE_GROUPS, n)
        detected_protected = ["gender", "race", "age_group"]

    # Synthetic target (not a real model feature — used only for cartography baseline)
    probe_target = "_probe_outcome"
    numeric_cols = [k for k, v in data.items() if k in model_feature_cols and isinstance(v[0], (int, np.integer))]
    if not numeric_cols:
        numeric_cols = [k for k, v in data.items() if isinstance(v[0], (int, np.integer))]
    if numeric_cols:
        score_col = numeric_cols[0]
        data[probe_target] = (
            (np.array(data[score_col]) > np.array(data[score_col]).mean()).astype(int)
        )
    else:
        data[probe_target] = rng.integers(0, 2, n).astype(int)

    df = pd.DataFrame(data)
    return df, df.to_csv(index=False), detected_protected, probe_target, model_feature_cols
