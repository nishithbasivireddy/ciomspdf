from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CheckboxDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent

        if split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                f"Unsupported split: {split}"
            )

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            rows = list(csv.DictReader(file))

        self.rows = [
            row
            for row in rows
            if row["split"] == split
        ]

        if not self.rows:
            raise ValueError(
                f"No checkbox samples for split: {split}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = self.root / row["image_path"]

        with Image.open(image_path) as image:
            grayscale = image.convert("L").resize(
                (32, 32),
                Image.Resampling.BILINEAR,
            )

            array = np.asarray(
                grayscale,
                dtype=np.float32,
            )

        array = 1.0 - (array / 255.0)

        image_tensor = torch.from_numpy(
            array
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),
            "class_name": row["class_name"],
            "image_path": str(image_path),
            "group_id": row["group_id"],
        }
