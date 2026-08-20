from __future__ import annotations

import torch

from src.ml.crnn.vocabulary import INDEX_TO_CHAR


BLANK_INDEX = 0


def greedy_ctc_decode(logits: torch.Tensor) -> list[str]:
    if logits.ndim != 3:
        raise ValueError(
            "Expected logits shaped as batch, time, classes."
        )

    predicted = logits.argmax(dim=2)
    decoded_texts = []

    for sequence in predicted:
        characters = []
        previous_index = None

        for item in sequence:
            index = int(item.item())

            if index != BLANK_INDEX and index != previous_index:
                characters.append(
                    INDEX_TO_CHAR.get(index, "")
                )

            previous_index = index

        decoded_texts.append("".join(characters))

    return decoded_texts


def normalize_for_metric(text: str) -> str:
    return " ".join(
        str(text).strip().casefold().split()
    )


def levenshtein_distance(reference, prediction) -> int:
    previous_row = list(range(len(prediction) + 1))

    for reference_index, reference_item in enumerate(
        reference,
        start=1,
    ):
        current_row = [reference_index]

        for prediction_index, prediction_item in enumerate(
            prediction,
            start=1,
        ):
            insertion = current_row[prediction_index - 1] + 1
            deletion = previous_row[prediction_index] + 1

            substitution = (
                previous_row[prediction_index - 1]
                + int(reference_item != prediction_item)
            )

            current_row.append(
                min(insertion, deletion, substitution)
            )

        previous_row = current_row

    return previous_row[-1]


def character_error_rate(
    reference: str,
    prediction: str,
) -> float:
    reference = normalize_for_metric(reference)
    prediction = normalize_for_metric(prediction)

    if not reference:
        return 0.0 if not prediction else 1.0

    distance = levenshtein_distance(
        list(reference),
        list(prediction),
    )

    return distance / len(reference)


def word_error_rate(
    reference: str,
    prediction: str,
) -> float:
    reference_words = normalize_for_metric(
        reference
    ).split()

    prediction_words = normalize_for_metric(
        prediction
    ).split()

    if not reference_words:
        return 0.0 if not prediction_words else 1.0

    distance = levenshtein_distance(
        reference_words,
        prediction_words,
    )

    return distance / len(reference_words)
