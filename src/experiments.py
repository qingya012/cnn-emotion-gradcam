"""Helpers for storing and discovering lightweight training experiments.

Each experiment is identified by a filesystem-safe name and stored under
``results/<experiment>/``. A complete, discoverable experiment contains:

* ``config.json``: the architecture key and training configuration used for
  the run, plus a human-readable description and optional migration notes.
* ``history.json``: epoch-level losses/accuracies and derived best-epoch
  metrics. Metrics that were never collected are represented by ``null``.
* ``model.pth``: the best validation checkpoint.
* ``best_y_true.npy`` and ``best_y_pred.npy``: labels and predictions from
  the epoch selected by the run's validation checkpoint criterion.
"""

import json
import re
from pathlib import Path

import numpy as np


CONFIG_FILENAME = "config.json"
HISTORY_FILENAME = "history.json"
CHECKPOINT_FILENAME = "model.pth"
TRUE_LABELS_FILENAME = "best_y_true.npy"
PREDICTIONS_FILENAME = "best_y_pred.npy"

_EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def get_experiment_dir(experiment: str, root: str | Path = ".") -> Path:
    """Return results/<experiment> after validating the directory name."""
    if not _EXPERIMENT_NAME.fullmatch(experiment):
        raise ValueError(
            "Experiment names must start with a letter or number and contain "
            "only letters, numbers, dots, underscores, or hyphens."
        )
    return Path(root) / "results" / experiment


def save_config(config: dict, root: str | Path = ".") -> Path:
    experiment = config.get("experiment")
    if not experiment:
        raise ValueError("Experiment config must include an 'experiment' field.")
    path = get_experiment_dir(experiment, root) / CONFIG_FILENAME
    _write_json(path, config)
    return path


def save_history(experiment: str, history: dict, root: str | Path = ".") -> Path:
    path = get_experiment_dir(experiment, root) / HISTORY_FILENAME
    _write_json(path, history)
    return path


def load_config(experiment: str, root: str | Path = ".") -> dict:
    path = get_experiment_dir(experiment, root) / CONFIG_FILENAME
    return _read_json(path, experiment)


def load_history(experiment: str, root: str | Path = ".") -> dict:
    path = get_experiment_dir(experiment, root) / HISTORY_FILENAME
    return _read_json(path, experiment)


def load_predictions(experiment: str, root: str | Path = "."):
    results = get_experiment_dir(experiment, root)
    y_true_path = results / TRUE_LABELS_FILENAME
    y_pred_path = results / PREDICTIONS_FILENAME
    missing = [
        path.name for path in (y_true_path, y_pred_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {results}. The experiment does not have "
            "saved best-epoch predictions."
        )
    return np.load(y_true_path), np.load(y_pred_path)


def list_experiments(root: str | Path = ".") -> list[str]:
    """List complete experiments, ignoring legacy or partial directories."""
    results_root = Path(root) / "results"
    if not results_root.exists():
        return []

    experiments = []
    for path in results_root.iterdir():
        if not (
            path.is_dir()
            and (path / CONFIG_FILENAME).is_file()
            and (path / HISTORY_FILENAME).is_file()
        ):
            continue
        try:
            get_experiment_dir(path.name, root)
            config = _read_json(path / CONFIG_FILENAME, path.name)
            _read_json(path / HISTORY_FILENAME, path.name)
        except (ValueError, json.JSONDecodeError):
            continue
        if config.get("experiment") == path.name:
            experiments.append(path.name)
    return sorted(experiments)


def experiment_summary_records(root: str | Path = ".") -> list[dict]:
    records = []
    for experiment in list_experiments(root):
        config = load_config(experiment, root)
        history = load_history(experiment, root)
        records.append(
            {
                "Experiment": experiment,
                "Architecture": config.get("architecture"),
                "Main Change": config.get("description"),
                "Best Val Accuracy": history.get("best_validation_accuracy"),
                "Epochs": config.get("epochs"),
            }
        )
    return records


def _read_json(path: Path, experiment: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Experiment {experiment!r} is incomplete or unknown."
        )
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write("\n")
    temporary_path.replace(path)
