from __future__ import annotations

from typing import Any


def normalize_value(value: Any) -> str:
    if value is None:
        return ""

    normalized = str(value)
    normalized = normalized.replace("\r", " ")
    normalized = normalized.replace("\n", " ")
    normalized = " ".join(normalized.split())

    return normalized.strip().casefold()


def safe_divide(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def evaluate_extraction(
    reference: dict[str, Any],
    prediction: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    field_results = []

    exact_matches = 0
    incorrect_values = 0
    missing_values = 0
    unexpected_values = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for field, expected_value in reference.items():
        actual_value = prediction.get(field, "")

        expected_normalized = normalize_value(
            expected_value
        )

        actual_normalized = normalize_value(
            actual_value
        )

        expected_present = bool(
            expected_normalized
        )

        actual_present = bool(
            actual_normalized
        )

        is_exact_match = (
            expected_normalized
            == actual_normalized
        )

        if is_exact_match:
            exact_matches += 1

            if expected_present:
                status = "Correct"
                true_positive += 1
            else:
                status = "Correct Blank"
                true_negative += 1

        elif expected_present and not actual_present:
            status = "Missing"
            missing_values += 1
            false_negative += 1

        elif not expected_present and actual_present:
            status = "Unexpected"
            unexpected_values += 1
            false_positive += 1

        else:
            status = "Incorrect"
            incorrect_values += 1

            # A wrong non-empty value both invents an
            # incorrect value and misses the expected
            # value under strict extraction scoring.
            false_positive += 1
            false_negative += 1

        field_results.append(
            {
                "field": field,
                "expected": expected_value,
                "actual": actual_value,
                "expected_normalized": (
                    expected_normalized
                ),
                "actual_normalized": (
                    actual_normalized
                ),
                "status": status,
                "match": is_exact_match,
            }
        )

    total_fields = len(reference)

    mismatched_fields = (
        total_fields - exact_matches
    )

    precision = safe_divide(
        true_positive,
        true_positive + false_positive,
    )

    recall = safe_divide(
        true_positive,
        true_positive + false_negative,
    )

    f1_score = safe_divide(
        2 * precision * recall,
        precision + recall,
    )

    metrics = {
        "accuracy": safe_divide(
            exact_matches,
            total_fields,
        ),
        "field_agreement": safe_divide(
            exact_matches,
            total_fields,
        ),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "confusion_matrix": [
            [
                true_positive,
                false_negative,
            ],
            [
                false_positive,
                true_negative,
            ],
        ],
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "total_fields": total_fields,
        "matched_fields": exact_matches,
        "missed_or_wrong_fields": (
            mismatched_fields
        ),
        "incorrect_values": incorrect_values,
        "missing_values": missing_values,
        "unexpected_values": unexpected_values,
    }

    return metrics, field_results
