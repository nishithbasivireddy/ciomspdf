from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def normalize_value(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\r", " ")
    value = value.replace("\n", " ")
    value = " ".join(value.split())
    return value.strip().lower()


def evaluate_extraction(reference, prediction):
    y_true = []
    y_pred = []
    field_results = []

    for field, expected_value in reference.items():
        actual_value = prediction.get(field, "")

        expected_norm = normalize_value(expected_value)
        actual_norm = normalize_value(actual_value)

        is_match = expected_norm == actual_norm

        y_true.append(1)
        y_pred.append(1 if is_match else 0)

        if is_match:
            status = "Correct"
        elif actual_norm == "":
            status = "Missing"
        else:
            status = "Wrong"

        field_results.append(
            {
                "field": field,
                "expected": expected_value,
                "actual": actual_value,
                "status": status,
                "match": is_match,
            }
        )

    matched = sum(y_pred)
    total = len(y_true)
    missed_or_wrong = total - matched

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[1, 0]).tolist(),
        "total_fields": total,
        "matched_fields": matched,
        "missed_or_wrong_fields": missed_or_wrong,
    }

    return metrics, field_results
