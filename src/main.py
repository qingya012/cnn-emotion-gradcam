"""Train, save, and load CNN experiments for the FER2013 dataset.

This module contains the FER2013 dataset adapter, CNN architectures, 
their shared ``MODELS`` registry, training and validation loops, and helpers 
used by the exploration notebook. Importing the module only defines these 
components; training starts only when this file is run as a script.

Architecture and experiment are intentionally separate concepts:

* ``--model`` chooses the neural-network architecture from ``MODELS``.
* ``--experiment`` gives one training run a unique name. Its artifacts are
  written to ``results/<experiment>/``.

For example:

``python src/main.py --experiment simple_run_2 --model simple --description "Repeat baseline"``

The command and its options mean:

* ``python src/main.py`` runs the training entry point in this file.
* ``--experiment simple_run_2`` names the run and its results directory.
* ``--model simple`` selects ``SimpleCNN``; use ``vgg`` for ``VGGStyleCNN``.
* ``--description "..."`` records a human-readable explanation in
  ``config.json`` and the notebook summary. It does not alter training.
* ``--epochs``, ``--batch-size``, and ``--lr`` optionally override their
  defaults of 10, 64, and 0.001.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from experiments import (
    CHECKPOINT_FILENAME,
    PREDICTIONS_FILENAME,
    TRUE_LABELS_FILENAME,
    get_experiment_dir,
    load_config,
    load_history as load_experiment_history,
    load_predictions as load_experiment_predictions,
    save_config,
    save_history as save_experiment_history,
)


class FERDataset(Dataset):
    class_names = (
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    )

    def __init__(self, csv_file, split="Training"):
        df = pd.read_csv(csv_file)
        self.df = df[df["Usage"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        label = int(row["emotion"])

        pixels = row["pixels"].split()
        pixels = np.array(pixels, dtype=np.float32).reshape(48, 48) / 255.0

        img = torch.tensor(pixels).unsqueeze(0)

        return img, label


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # input: 48x48
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 24x24

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 12x12
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 12 * 12, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=7),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGGStyleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # VGG-style: 3 blocks of (Conv3x3 → ReLU → Conv3x3 → ReLU → MaxPool2x2)
        self.features = nn.Sequential(
            # Block 1: 48x48 → 24x24
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 24x24 → 12x12
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 12x12 → 6x6
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 6 * 6, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=7),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGGStyleCNNV1(nn.Module):
    """Three-block VGG-style model with ReLU followed by BatchNorm per conv."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 48x48 → 24x24
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 24x24 → 12x12
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 12x12 → 6x6
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=256 * 6 * 6, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=7),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class VGGStyleCNNV2(nn.Module):
    """Four-block VGG-style model extending V1 with a 512-channel block."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 48x48 → 24x24
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 24x24 → 12x12
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=128),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 12x12 → 6x6
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=256),
            nn.MaxPool2d(kernel_size=2),

            # Block 4: 6x6 → 3x3
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 3 * 3, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=7),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# CLI name → architecture + Grad-CAM target (last Conv2d in features)
MODELS = {
    "simple": {
        "class": SimpleCNN,
        "gradcam_layer_index": 3,
    },
    "vgg": {
        "class": VGGStyleCNN,
        "gradcam_layer_index": 12,
    },
    "vgg_v1": {
        "class": VGGStyleCNNV1,
        "gradcam_layer_index": 17,
    },
    "vgg_v2": {
        "class": VGGStyleCNNV2,
        "gradcam_layer_index": 24,
    },
}


def get_results_dir(experiment: str, root: str | Path = ".") -> Path:
    return get_experiment_dir(experiment, root)


def build_model(model_name: str) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {list(MODELS)}")
    return MODELS[model_name]["class"]()


def get_gradcam_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {list(MODELS)}")
    return model.features[MODELS[model_name]["gradcam_layer_index"]]


def load_history(experiment: str, root: str | Path = "."):
    return load_experiment_history(experiment, root)


def load_predictions(experiment: str, root: str | Path = "."):
    return load_experiment_predictions(experiment, root)


def get_device() -> torch.device:
    """Prefer CUDA, then Apple Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_model(experiment: str, device, root: str | Path = "."):
    config = load_config(experiment, root)
    model_name = config["model"]
    results = get_results_dir(experiment, root)
    checkpoint = results / CHECKPOINT_FILENAME
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. Train with:\n"
            f"  python src/main.py --experiment {experiment} --model {model_name}"
        )
    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        average_loss = running_loss / len(loader)
        accuracy = correct / total
        return average_loss, accuracy


def get_predictions(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FER emotion CNN")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Unique run name (results saved under results/<experiment>/)",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS),
        default="simple",
        help="Architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--description",
        default="",
        help="Short description of the change being tested",
    )
    args = parser.parse_args()

    device = get_device()
    root = Path(".")
    out_dir = get_results_dir(args.experiment, root)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(
        {
            "experiment": args.experiment,
            "model": args.model,
            "architecture": MODELS[args.model]["class"].__name__,
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "model_options": {},
            "training_options": {
                "loss": "CrossEntropyLoss",
                "training_shuffle": True,
                "training_split": "Training",
                "validation_split": "PublicTest",
            },
            "augmentation": {
                "enabled": False,
                "transforms": [],
            },
            "description": args.description,
            "migration": None,
        },
        root,
    )

    train_dataset = FERDataset("data/fer2013.csv", split="Training")
    val_dataset = FERDataset("data/fer2013.csv", split="PublicTest")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_accuracy = 0.0
    best_y_true = None
    best_y_pred = None
    train_losses = []
    val_losses = []
    val_accuracies = []

    print(
        f"Training experiment={args.experiment}, model={args.model} on {device}; "
        f"saving to {out_dir}/"
    )

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        y_pred, y_true = get_predictions(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_y_true = y_true
            best_y_pred = y_pred
            torch.save(model.state_dict(), out_dir / CHECKPOINT_FILENAME)

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    print(f"Best Val Accuracy: {best_val_accuracy:.4f}")

    np.save(out_dir / TRUE_LABELS_FILENAME, np.array(best_y_true))
    np.save(out_dir / PREDICTIONS_FILENAME, np.array(best_y_pred))
    save_experiment_history(
        args.experiment,
        {
            "train_loss": train_losses,
            "validation_loss": val_losses,
            "train_accuracy": None,
            "validation_accuracy": val_accuracies,
            "best_validation_accuracy": best_val_accuracy,
            "best_epoch": int(np.argmax(val_accuracies)) + 1,
            "test_accuracy": None,
        },
        root,
    )
