from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.ml.crnn.vocabulary import CHAR_TO_INDEX


IMAGE_HEIGHT = 32
IMAGE_WIDTH = 512


class CRNNTextDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.split = split

        if split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                f"Unsupported split: {split}"
            )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
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
                f"No samples found for split: {split}"
            )

        for row in self.rows:
            image_path = self.root / row["image_path"]

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Training image missing: {image_path}"
                )

            label = row["text"]

            unsupported = [
                character
                for character in label
                if character not in CHAR_TO_INDEX
            ]

            if unsupported:
                raise ValueError(
                    f"Unsupported characters in {label!r}: "
                    f"{unsupported}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = self.root / row["image_path"]
        label_text = row["text"]

        with Image.open(image_path) as image:
            image = image.convert("L")
            image = image.resize(
                (IMAGE_WIDTH, IMAGE_HEIGHT),
                Image.Resampling.BILINEAR,
            )

            image_array = np.asarray(
                image,
                dtype=np.float32,
            )

        image_array = image_array / 255.0
        image_array = 1.0 - image_array

        image_tensor = torch.from_numpy(
            image_array
        ).unsqueeze(0)

        target_indices = [
            CHAR_TO_INDEX[character]
            for character in label_text
        ]

        target_tensor = torch.tensor(
            target_indices,
            dtype=torch.long,
        )

        return {
            "image": image_tensor,
            "target": target_tensor,
            "target_length": len(target_indices),
            "text": label_text,
            "image_path": str(image_path),
            "group_id": row["group_id"],
        }


def ctc_collate_fn(batch):
    images = torch.stack(
        [item["image"] for item in batch],
        dim=0,
    )

    targets = torch.cat(
        [item["target"] for item in batch],
        dim=0,
    )

    target_lengths = torch.tensor(
        [
            item["target_length"]
            for item in batch
        ],
        dtype=torch.long,
    )

    texts = [
        item["text"]
        for item in batch
    ]

    image_paths = [
        item["image_path"]
        for item in batch
    ]

    group_ids = [
        item["group_id"]
        for item in batch
    ]

    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": texts,
        "image_paths": image_paths,
        "group_ids": group_ids,
    }
