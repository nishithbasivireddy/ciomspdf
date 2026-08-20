from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.ml.crnn.cioms_domain_dataset import (
    CIOMSDomainDataset,
    cioms_collate_fn,
)
from src.ml.crnn.decoder import (
    character_error_rate,
    greedy_ctc_decode,
    normalize_for_metric,
    word_error_rate,
)
from src.ml.crnn.model import CRNNTextRecognizer
from src.ml.crnn.vocabulary import VOCABULARY


SEED = 20260820

DEFAULT_MANIFEST = (
    "data/ml_training/cioms_rendered/manifest.csv"
)

BASE_MODEL_PATH = (
    "models/custom_ml/crnn_text_recognizer.pt"
)

DEFAULT_MODEL_PATH = (
    "models/custom_ml/crnn_cioms_finetuned.pt"
)

DEFAULT_METADATA_PATH = (
    "models/custom_ml/crnn_cioms_finetuned_metadata.json"
)

DEFAULT_HISTORY_PATH = (
    "models/custom_ml/crnn_cioms_finetuning_history.json"
)

DEFAULT_TEST_REPORT_PATH = (
    "models/custom_ml/crnn_cioms_finetuned_test_report.json"
)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loader(
    manifest_path: str,
    split: str,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CIOMSDomainDataset(
        manifest_path=manifest_path,
        split=split,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=cioms_collate_fn,
        generator=generator,
        pin_memory=False,
    )


def calculate_ctc_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    loss_function: nn.CTCLoss,
) -> torch.Tensor:
    log_probabilities = logits.log_softmax(dim=2)
    log_probabilities = log_probabilities.permute(
        1,
        0,
        2,
    )

    time_steps = log_probabilities.shape[0]
    batch_size = log_probabilities.shape[1]

    input_lengths = torch.full(
        size=(batch_size,),
        fill_value=time_steps,
        dtype=torch.long,
        device="cpu",
    )

    return loss_function(
        log_probabilities,
        targets,
        input_lengths,
        target_lengths,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.CTCLoss,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    batch_count = 0

    for batch in loader:
        images = batch["images"].to(device)

        targets = batch["targets"].to(device)

        target_lengths = batch[
            "target_lengths"
        ].to(dtype=torch.long, device="cpu")

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = calculate_ctc_loss(
            logits=logits,
            targets=targets,
            target_lengths=target_lengths,
            loss_function=loss_function,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite CTC loss encountered."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        total_loss += float(loss.item())
        batch_count += 1

    if batch_count == 0:
        raise RuntimeError(
            "Training loader produced no batches."
        )

    return total_loss / batch_count


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.CTCLoss,
    device: torch.device,
    include_predictions: bool,
) -> dict:
    model.eval()

    total_loss = 0.0
    batch_count = 0

    total_character_error = 0.0
    total_word_error = 0.0
    exact_matches = 0
    sample_count = 0
    prediction_rows = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["images"].to(device)

            targets = batch["targets"].to(device)

            target_lengths = batch[
                "target_lengths"
            ].to(dtype=torch.long, device="cpu")

            logits = model(images)

            loss = calculate_ctc_loss(
                logits=logits,
                targets=targets,
                target_lengths=target_lengths,
                loss_function=loss_function,
            )

            predictions = greedy_ctc_decode(
                logits.detach().cpu()
            )

            total_loss += float(loss.item())
            batch_count += 1

            for reference, prediction, image_path in zip(
                batch["texts"],
                predictions,
                batch["image_paths"],
            ):
                cer = character_error_rate(
                    reference,
                    prediction,
                )

                wer = word_error_rate(
                    reference,
                    prediction,
                )

                is_exact = (
                    normalize_for_metric(reference)
                    == normalize_for_metric(prediction)
                )

                total_character_error += cer
                total_word_error += wer
                exact_matches += int(is_exact)
                sample_count += 1

                if include_predictions:
                    prediction_rows.append(
                        {
                            "image_path": image_path,
                            "reference": reference,
                            "prediction": prediction,
                            "exact_match": is_exact,
                            "character_error_rate": cer,
                            "word_error_rate": wer,
                        }
                    )

    if batch_count == 0 or sample_count == 0:
        raise RuntimeError(
            "Evaluation loader produced no samples."
        )

    return {
        "loss": total_loss / batch_count,
        "sample_count": sample_count,
        "exact_match_count": exact_matches,
        "exact_match_accuracy": (
            exact_matches / sample_count
        ),
        "mean_character_error_rate": (
            total_character_error / sample_count
        ),
        "mean_word_error_rate": (
            total_word_error / sample_count
        ),
        "predictions": prediction_rows,
    }


def save_model_checkpoint(
    model: nn.Module,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        path,
    )


def load_model_checkpoint(
    model: nn.Module,
    model_path: str,
    device: torch.device,
) -> None:
    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)


def save_json(
    data: dict | list,
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate a custom CRNN "
            "text-recognition model using CTC loss."
        )
    )

    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "Epoch count must be at least one."
        )

    if args.batch_size < 1:
        raise ValueError(
            "Batch size must be at least one."
        )

    set_reproducible_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Vocabulary classes:", len(VOCABULARY))
    print("Epochs:", args.epochs)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.learning_rate)

    train_loader = create_loader(
        manifest_path=args.manifest,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
    )

    validation_loader = create_loader(
        manifest_path=args.manifest,
        split="validation",
        batch_size=args.batch_size,
        shuffle=False,
    )

    test_loader = create_loader(
        manifest_path=args.manifest,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = CRNNTextRecognizer(
        number_of_classes=len(VOCABULARY),
        hidden_size=128,
    ).to(device)

    base_model_path = Path(BASE_MODEL_PATH)

    if not base_model_path.is_file():
        raise FileNotFoundError(
            f"Base CRNN checkpoint is missing: "
            f"{base_model_path}"
        )

    base_state = torch.load(
        base_model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(base_state)

    print(
        "Loaded base checkpoint:",
        base_model_path,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Total parameters:", parameter_count)
    print(
        "Trainable parameters:",
        trainable_parameter_count,
    )

    loss_function = nn.CTCLoss(
        blank=0,
        reduction="mean",
        zero_infinity=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )

    history = []
    best_validation_cer = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        training_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
        )

        validation_metrics = evaluate_model(
            model=model,
            loader=validation_loader,
            loss_function=loss_function,
            device=device,
            include_predictions=False,
        )

        epoch_record = {
            "epoch": epoch,
            "training_loss": training_loss,
            "validation_loss": validation_metrics[
                "loss"
            ],
            "validation_exact_match_accuracy": (
                validation_metrics[
                    "exact_match_accuracy"
                ]
            ),
            "validation_character_error_rate": (
                validation_metrics[
                    "mean_character_error_rate"
                ]
            ),
            "validation_word_error_rate": (
                validation_metrics[
                    "mean_word_error_rate"
                ]
            ),
        }

        history.append(epoch_record)

        print()
        print(
            f"Epoch {epoch}/{args.epochs}"
        )
        print(
            f"Training loss: "
            f"{training_loss:.6f}"
        )
        print(
            f"Validation loss: "
            f"{validation_metrics['loss']:.6f}"
        )
        print(
            f"Validation exact match: "
            f"{validation_metrics['exact_match_accuracy']:.4f}"
        )
        print(
            f"Validation CER: "
            f"{validation_metrics['mean_character_error_rate']:.4f}"
        )
        print(
            f"Validation WER: "
            f"{validation_metrics['mean_word_error_rate']:.4f}"
        )

        current_cer = validation_metrics[
            "mean_character_error_rate"
        ]

        if current_cer < best_validation_cer:
            best_validation_cer = current_cer
            best_epoch = epoch
            epochs_without_improvement = 0

            save_model_checkpoint(
                model=model,
                output_path=args.model_path,
            )

            print(
                "Best checkpoint saved:",
                args.model_path,
            )

        else:
            epochs_without_improvement += 1

        save_json(
            history,
            DEFAULT_HISTORY_PATH,
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping activated because "
                "validation CER did not improve."
            )
            break

    if not Path(args.model_path).exists():
        raise RuntimeError(
            "No trained model checkpoint was saved."
        )

    load_model_checkpoint(
        model=model,
        model_path=args.model_path,
        device=device,
    )

    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        loss_function=loss_function,
        device=device,
        include_predictions=True,
    )

    metadata = {
        "model_type": (
            "CIOMS-domain fine-tuned CNN plus "
            "bidirectional LSTM text recognizer "
            "trained with CTC loss"
        ),
        "task": (
            "Single-line CIOMS rendered field-crop transcription"
        ),
        "input_shape": [
            1,
            32,
            512,
        ],
        "number_of_classes": len(VOCABULARY),
        "blank_index": 0,
        "hidden_size": 128,
        "total_parameters": parameter_count,
        "trainable_parameters": (
            trainable_parameter_count
        ),
        "seed": SEED,
        "device_used_for_training": str(device),
        "requested_epochs": args.epochs,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_character_error_rate": (
            best_validation_cer
        ),
        "train_samples": len(
            train_loader.dataset
        ),
        "validation_samples": len(
            validation_loader.dataset
        ),
        "test_samples": len(
            test_loader.dataset
        ),
        "test_exact_match_accuracy": (
            test_metrics[
                "exact_match_accuracy"
            ]
        ),
        "test_mean_character_error_rate": (
            test_metrics[
                "mean_character_error_rate"
            ]
        ),
        "test_mean_word_error_rate": (
            test_metrics[
                "mean_word_error_rate"
            ]
        ),
        "model_path": args.model_path,
        "integrity_statement": (
            "The model was fine-tuned from the verified generic "
            "CRNN using synthetic CIOMS-rendered crop images. "
            "The protected real evaluation PDF, Plain Python, "
            "OCR, and LLM outputs were excluded from training."
        ),
        "known_limitation": (
            "The trained recognizer handles single-line "
            "text crops. Multiline CIOMS regions require "
            "line segmentation before recognition."
        ),
    }

    save_json(
        metadata,
        DEFAULT_METADATA_PATH,
    )

    save_json(
        test_metrics,
        DEFAULT_TEST_REPORT_PATH,
    )

    print()
    print("TRAINING AND TEST EVALUATION COMPLETED")
    print("--------------------------------------")
    print("Best epoch:", best_epoch)
    print(
        "Best validation CER:",
        best_validation_cer,
    )
    print(
        "Test exact match:",
        test_metrics["exact_match_accuracy"],
    )
    print(
        "Test mean CER:",
        test_metrics[
            "mean_character_error_rate"
        ],
    )
    print(
        "Test mean WER:",
        test_metrics[
            "mean_word_error_rate"
        ],
    )
    print("Model saved:", args.model_path)
    print(
        "Metadata saved:",
        DEFAULT_METADATA_PATH,
    )
    print(
        "Test report saved:",
        DEFAULT_TEST_REPORT_PATH,
    )


if __name__ == "__main__":
    main()
