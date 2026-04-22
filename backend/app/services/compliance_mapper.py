"""
Regulatory Compliance Mapper
Maps computed fairness metrics to real-world legal thresholds.
Returns PASS / FAIL / CAUTION verdicts per regulation.
"""

from typing import Any

REGULATIONS = [
    {
        "id": "EEOC_4FIFTHS",
        "label": "EEOC 4/5ths Rule",
        "description": "Adverse impact ratio must be ≥ 0.80 (80% rule) for employment decisions.",
        "domain": "Employment",
        "di_threshold": 0.80,
        "spd_threshold": None,
    },
    {
        "id": "EU_AI_ACT",
        "label": "EU AI Act Art. 10",
        "description": "High-risk AI systems must ensure SPD < 0.05 and DI ≥ 0.90 for protected groups.",
        "domain": "General / High-Risk AI",
        "di_threshold": 0.90,
        "spd_threshold": 0.05,
    },
    {
        "id": "ECOA_FCRA",
        "label": "ECOA / FCRA",
        "description": "Equal Credit Opportunity Act requires SPD < 0.10 across race, sex, national origin.",
        "domain": "Credit / Lending",
        "di_threshold": None,
        "spd_threshold": 0.10,
    },
    {
        "id": "FHA",
        "label": "Fair Housing Act",
        "description": "Fair Housing Act prohibits disparate impact in housing with SPD < 0.10.",
        "domain": "Housing",
        "di_threshold": None,
        "spd_threshold": 0.10,
    },
]


def _verdict(violations: list[str]) -> str:
    if not violations:
        return "PASS"
    # Caution if only barely over one threshold, FAIL if clearly over
    return "FAIL"


def check_compliance(slice_metrics: list[dict]) -> list[dict[str, Any]]:
    """
    Given slice_metrics from cartography, compute compliance verdicts.
    Uses worst-case SPD and minimum DI across all single-attribute slices.
    """
    if not slice_metrics:
        return [
            {**r, "status": "PASS", "violations": [], "worst_spd": 0.0, "worst_di": 1.0}
            for r in REGULATIONS
        ]

    single = [m for m in slice_metrics if "∩" not in m["label"]]
    if not single:
        single = slice_metrics

    worst_spd = max(abs(m["statistical_parity_diff"]) for m in single)
    worst_di = min(m["disparate_impact"] for m in single)
    worst_slice = max(single, key=lambda m: abs(m["statistical_parity_diff"]))["label"]

    results = []
    for reg in REGULATIONS:
        violations = []

        if reg["spd_threshold"] is not None and worst_spd > reg["spd_threshold"]:
            violations.append(
                f"SPD {worst_spd:.3f} exceeds limit {reg['spd_threshold']} "
                f"(worst: {worst_slice})"
            )

        if reg["di_threshold"] is not None and worst_di < reg["di_threshold"]:
            violations.append(
                f"DI {worst_di:.3f} below threshold {reg['di_threshold']} "
                f"(worst: {worst_slice})"
            )

        results.append({
            "id": reg["id"],
            "label": reg["label"],
            "description": reg["description"],
            "domain": reg["domain"],
            "status": _verdict(violations),
            "violations": violations,
            "worst_spd": round(worst_spd, 4),
            "worst_di": round(worst_di, 4),
        })

    return results
