import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


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
}


def get_results_dir(model_name: str, root: str | Path = ".") -> Path:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {list(MODELS)}")
    return Path(root) / "results" / model_name


def build_model(model_name: str) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {list(MODELS)}")
    return MODELS[model_name]["class"]()


def get_gradcam_layer(model: nn.Module, model_name: str) -> nn.Module:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}. Choose from: {list(MODELS)}")
    return model.features[MODELS[model_name]["gradcam_layer_index"]]


def load_history(model_name: str, root: str | Path = "."):
    results = get_results_dir(model_name, root)
    required = ("train_losses.npy", "val_losses.npy", "val_accuracies.npy")
    missing = [name for name in required if not (results / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {results}. Train with:\n"
            f"  python src/main.py --model {model_name}"
        )
    return {
        "train_losses": np.load(results / "train_losses.npy"),
        "val_losses": np.load(results / "val_losses.npy"),
        "val_accuracies": np.load(results / "val_accuracies.npy"),
    }


def load_predictions(model_name: str, root: str | Path = "."):
    results = get_results_dir(model_name, root)
    y_true_path = results / "best_y_true.npy"
    y_pred_path = results / "best_y_pred.npy"
    if not y_true_path.exists() or not y_pred_path.exists():
        raise FileNotFoundError(
            f"Missing prediction files under {results}. Train with:\n"
            f"  python src/main.py --model {model_name}"
        )
    return np.load(y_true_path), np.load(y_pred_path)


def get_device() -> torch.device:
    """Prefer CUDA, then Apple Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_model(model_name: str, device, root: str | Path = "."):
    results = get_results_dir(model_name, root)
    checkpoint = results / "best_model.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"{checkpoint} not found. Train with:\n"
            f"  python src/main.py --model {model_name}"
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
        "--model",
        choices=list(MODELS),
        default="simple",
        help="Architecture to train (results saved under results/<model>/)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    device = get_device()
    root = Path(".")
    out_dir = get_results_dir(args.model, root)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Training model={args.model} on {device}; saving to {out_dir}/")

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
            torch.save(model.state_dict(), out_dir / "best_model.pth")

        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

    print(f"Best Val Accuracy: {best_val_accuracy:.4f}")

    np.save(out_dir / "best_y_true.npy", np.array(best_y_true))
    np.save(out_dir / "best_y_pred.npy", np.array(best_y_pred))
    np.save(out_dir / "train_losses.npy", np.array(train_losses))
    np.save(out_dir / "val_losses.npy", np.array(val_losses))
    np.save(out_dir / "val_accuracies.npy", np.array(val_accuracies))
