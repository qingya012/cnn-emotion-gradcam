"""Evaluate the selected final V6 checkpoint once on FER2013 PrivateTest."""

import hashlib
import json
from pathlib import Path

import torch.nn as nn
from torch.utils.data import DataLoader

from experiments import CHECKPOINT_FILENAME, load_config, load_history
from main import (
    FERDataset,
    build_evaluation_transform,
    evaluate,
    get_device,
    get_results_dir,
    load_trained_model,
)


FINAL_EXPERIMENT = "vgg_v6_extended_lr_scheduler"
FINAL_EVALUATION_FILENAME = "final_evaluation.json"


def main():
    root = Path(__file__).resolve().parents[1]
    config = load_config(FINAL_EXPERIMENT, root)
    history = load_history(FINAL_EXPERIMENT, root)
    results_dir = get_results_dir(FINAL_EXPERIMENT, root)
    checkpoint = results_dir / CHECKPOINT_FILENAME

    if config["training_options"]["checkpoint_monitor"] != "validation_loss":
        raise ValueError("The selected V6 checkpoint was not chosen by validation loss.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Selected V6 checkpoint not found: {checkpoint}")

    device = get_device()
    model = load_trained_model(FINAL_EXPERIMENT, device, root)
    private_test_dataset = FERDataset(
        root / "data" / "fer2013.csv",
        split="PrivateTest",
        transform=build_evaluation_transform(),
    )
    private_test_loader = DataLoader(
        private_test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    private_test_loss, private_test_accuracy = evaluate(
        model,
        private_test_loader,
        nn.CrossEntropyLoss(),
        device,
    )

    evaluation = {
        "selected_experiment": FINAL_EXPERIMENT,
        "selected_model": config["architecture"],
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_selection": "minimum validation loss",
        "checkpoint_epoch": history["best_checkpoint_epoch"],
        "best_validation_loss": history["best_validation_loss"],
        "private_test_samples": len(private_test_dataset),
        "private_test_loss": private_test_loss,
        "private_test_accuracy": private_test_accuracy,
    }

    output_path = results_dir / FINAL_EVALUATION_FILENAME
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2, allow_nan=False)
        file.write("\n")

    print(f"Selected model: V6 — Extended LR Scheduler")
    print(f"Checkpoint: {checkpoint}")
    print(f"Checkpoint epoch: {history['best_checkpoint_epoch']}")
    print(f"PrivateTest Loss: {private_test_loss:.6f}")
    print(f"PrivateTest Accuracy: {private_test_accuracy:.6f}")
    print(f"Saved final evaluation: {output_path}")


if __name__ == "__main__":
    main()
