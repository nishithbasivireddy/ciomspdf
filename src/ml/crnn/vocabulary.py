from __future__ import annotations

import json
import string
from pathlib import Path

BLANK_TOKEN = "<BLANK>"

CHARACTERS = (
    string.ascii_letters
    + string.digits
    + " "
    + ".,:/\\-+()[]'%&@#"
)

VOCABULARY = [BLANK_TOKEN] + list(dict.fromkeys(CHARACTERS))

CHAR_TO_INDEX = {
    character: index
    for index, character in enumerate(VOCABULARY)
}

INDEX_TO_CHAR = {
    index: character
    for character, index in CHAR_TO_INDEX.items()
}


def save_vocabulary(
    output_path: str = "models/custom_ml/vocabulary.json",
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "blank_token": BLANK_TOKEN,
        "blank_index": 0,
        "characters": VOCABULARY,
    }

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    save_vocabulary()
    print(f"Vocabulary size: {len(VOCABULARY)}")
