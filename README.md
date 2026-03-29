# Emotion Recognition with CNN and Grad-CAM

## Overview

This project builds a convolutional neural network (CNN) for facial emotion recognition, with a focus on model interpretability.
In addition to predicting emotion labels, the system uses Grad-CAM to visualize which regions of the image contribute most to the model's decision.

The goal is not only to achieve classification performance, but also to better understand how the model behaves and why it makes certain predictions.

---

## Motivation

Traditional image classification models act as black boxes — they output predictions without explaining their reasoning.
This project aims to bridge that gap by integrating interpretability techniques, allowing users to inspect and analyze model decisions.

---

## Features

* CNN-based image classification
* Grad-CAM visualization for interpretability
* Training and evaluation pipeline
* Confusion matrix and sample prediction analysis
* Modular code structure for future extension

---

## Dataset

This project uses the **FER2013** dataset (Facial Expression Recognition).

Download from Kaggle:
https://www.kaggle.com/datasets/msambare/fer2013

After downloading, place the file in:

```
data/fer2013.csv
```

---

## Project Structure

```
project/
│
├── data/                # dataset (not included in repo)
├── notebooks/           # experiments / visualization
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── gradcam.py
│   └── utils.py
│
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   └── gradcam/
│
├── README.md
└── requirements.txt
```

---

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

### 1. Train the model

```
python src/train.py
```

### 2. Evaluate the model

```
python src/evaluate.py
```

### 3. Run prediction with Grad-CAM

```
python src/predict.py
```

---

## Interpretability (Grad-CAM)

Grad-CAM is used to highlight the regions in an image that most influence the model’s prediction.

For each input image, the system produces:

* Original image
* Heatmap
* Overlay visualization

This helps analyze:

* What features the model focuses on
* Why certain predictions are correct or incorrect

---

## Future Work

* Improve model architecture (e.g., ResNet)
* Add interactive visualization (web demo)
* Perform deeper error analysis
* Support multiple datasets
* Extend to multi-label or real-world data

---

## Notes

* Dataset is not included due to size limitations.
* Make sure to download and place it in the correct directory before running the code.

---

## License

This project is for educational and research purposes.
