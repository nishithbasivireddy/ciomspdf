from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.ml.crnn.real_pdf_inference import preprocess_line
from src.ml.crnn.vocabulary import CHAR_TO_INDEX


class CIOMSDomainDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.split = split

        allowed_splits = {
            "train",
            "validation",
            "test",
        }

        if split not in allowed_splits:
            raise ValueError(
                f"Unsupported split: {split}"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            all_rows = list(csv.DictReader(file))

        self.rows = [
            row
            for row in all_rows
            if row["split"] == split
        ]

        if not self.rows:
            raise ValueError(
                f"No manifest rows found for split: {split}"
            )

        for row in self.rows:
            image_path = (
                self.root / row["image_path"]
            )

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Crop image missing: {image_path}"
                )

            label = row["text"]

            if not label.strip():
                raise ValueError(
                    f"Blank label found for: {image_path}"
                )

            unsupported = sorted({
                character
                for character in label
                if character not in CHAR_TO_INDEX
            })

            if unsupported:
                raise ValueError(
                    f"Unsupported characters in "
                    f"{label!r}: {unsupported}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]

        image_path = (
            self.root / row["image_path"]
        )

        with Image.open(image_path) as image:
            image_tensor = preprocess_line(
                image.convert("RGB")
            ).squeeze(0)

        label = row["text"]

        target = torch.tensor(
            [
                CHAR_TO_INDEX[character]
                for character in label
            ],
            dtype=torch.long,
        )

        return {
            "image": image_tensor,
            "target": target,
            "target_length": len(label),
            "text": label,
            "image_path": str(image_path),
            "group_id": row["document_id"],
            "document_id": row["document_id"],
            "field": row["field"],
        }


def cioms_collate_fn(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    images = torch.stack(
        [
            item["image"]
            for item in batch
        ],
        dim=0,
    )

    targets = torch.cat(
        [
            item["target"]
            for item in batch
        ],
        dim=0,
    )

    target_lengths = torch.tensor(
        [
            item["target_length"]
            for item in batch
        ],
        dtype=torch.long,
    )

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": [
            item["text"]
            for item in batch
        ],
        "image_paths": [
            item["image_path"]
            for item in batch
        ],
        "group_ids": [
            item["group_id"]
            for item in batch
        ],
        "document_ids": [
            item["document_id"]
            for item in batch
        ],
        "fields": [
            item["field"]
            for item in batch
        ],
    }
