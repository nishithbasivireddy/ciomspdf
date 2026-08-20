from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.ml.checkbox.dataset import CheckboxDataset
from src.ml.checkbox.model import CheckboxCNN


SEED = 20260820
MANIFEST_PATH = "data/ml_training/checkbox/manifest.csv"
MODEL_PATH = Path("models/custom_ml/checkbox_cnn.pt")
HISTORY_PATH = Path("models/custom_ml/checkbox_training_history.json")
REPORT_PATH = Path("models/custom_ml/checkbox_test_report.json")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def create_loader(
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CheckboxDataset(
        MANIFEST_PATH,
        split,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    predictions = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = loss_function(logits, labels)
            predicted = logits.argmax(dim=1)

            total_loss += float(loss.item())
            correct += int(
                (predicted == labels).sum().item()
            )
            total += int(labels.numel())

            for reference, prediction, image_path in zip(
                labels.cpu().tolist(),
                predicted.cpu().tolist(),
                batch["image_path"],
            ):
                predictions.append(
                    {
                        "image_path": image_path,
                        "reference": reference,
                        "prediction": prediction,
                        "correct": (
                            reference == prediction
                        ),
                    }
                )

    return {
        "loss": total_loss / len(loader),
        "sample_count": total,
        "correct_count": correct,
        "accuracy": correct / total,
        "predictions": predictions,
    }


def main() -> None:
    set_seed()

    device = torch.device("cpu")
    epochs = 10
    batch_size = 32

    train_loader = create_loader(
        "train",
        batch_size,
        True,
    )

    validation_loader = create_loader(
        "validation",
        batch_size,
        False,
    )

    test_loader = create_loader(
        "test",
        batch_size,
        False,
    )

    model = CheckboxCNN().to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001,
    )

    best_accuracy = -1.0
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        training_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)
            loss = loss_function(
                logits,
                labels,
            )

            loss.backward()
            optimizer.step()

            training_loss += float(
                loss.item()
            )

        validation = evaluate(
            model,
            validation_loader,
            loss_function,
            device,
        )

        epoch_record = {
            "epoch": epoch,
            "training_loss": (
                training_loss
                / len(train_loader)
            ),
            "validation_loss": (
                validation["loss"]
            ),
            "validation_accuracy": (
                validation["accuracy"]
            ),
        }

        history.append(epoch_record)

        print()
        print(f"Epoch {epoch}/{epochs}")
        print(
            "Training loss:",
            round(
                epoch_record[
                    "training_loss"
                ],
                6,
            ),
        )
        print(
            "Validation loss:",
            round(
                validation["loss"],
                6,
            ),
        )
        print(
            "Validation accuracy:",
            round(
                validation["accuracy"],
                4,
            ),
        )

        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation[
                "accuracy"
            ]
            best_epoch = epoch

            MODEL_PATH.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

            print(
                "Best checkpoint saved:",
                MODEL_PATH,
            )

    HISTORY_PATH.write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True,
        )
    )

    test_report = evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )

    test_report["best_epoch"] = best_epoch
    test_report[
        "best_validation_accuracy"
    ] = best_accuracy

    REPORT_PATH.write_text(
        json.dumps(
            test_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("CHECKBOX TRAINING COMPLETED")
    print("---------------------------")
    print("Best epoch:", best_epoch)
    print(
        "Best validation accuracy:",
        best_accuracy,
    )
    print(
        "Test accuracy:",
        test_report["accuracy"],
    )
    print("Model saved:", MODEL_PATH)
    print("Test report:", REPORT_PATH)


if __name__ == "__main__":
    main()
