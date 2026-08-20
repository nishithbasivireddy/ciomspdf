from __future__ import annotations

import torch
from torch import nn


class CRNNTextRecognizer(nn.Module):
    def __init__(
        self,
        number_of_classes: int,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=(2, 1),
                stride=(2, 1),
            ),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.sequence_model = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            number_of_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)
        features = features.squeeze(2)
        features = features.permute(0, 2, 1)

        sequence_output, _ = self.sequence_model(features)
        logits = self.classifier(sequence_output)

        return logits
