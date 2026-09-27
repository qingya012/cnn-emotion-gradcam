# FER2013 Emotion Recognition with CNN and Grad-CAM

## Overview

This project trains and iteratively improves a convolutional neural network for facial-expression classification on FER2013. It develops a deeper VGG-style architecture with standard training techniques, analyzes the final model's predictions and systematic errors, and uses Grad-CAM to investigate which image regions contribute to class predictions and selected misclassifications.

The focus is model development and interpretability rather than leaderboard optimization.

## Dataset

The project uses the FER2013 dataset, containing 48 × 48 grayscale face images labeled as Angry, Disgust, Fear, Happy, Sad, Surprise, or Neutral.

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place the CSV at:

```text
data/fer2013.csv
```

The training split is used for fitting, PublicTest for validation and model selection, and PrivateTest only for the final evaluation and analysis of the selected model.

## Model Development

The model was developed through a sequence of controlled experiments:

- **V1 — BatchNorm:** added BatchNorm after convolutional layers.
- **V2 — VGG-style architecture:** expanded the network to four convolutional blocks with two convolutional layers per block (eight total), BatchNorm, and MaxPool.
- **V3 — Data augmentation:** added scaling, translation, rotation, and random erasing to training data.
- **V4 — Extended training:** increased the training duration after augmentation substantially slowed convergence.
- **V5 — LR scheduler:** added validation-loss-based learning-rate reduction.
- **V6 — Extended LR scheduler:** increased the maximum training window so the scheduler had more opportunity to affect convergence.

V6 was selected as the final configuration. Final analysis uses its best-validation-loss checkpoint, not its final-epoch weights.

- **PrivateTest accuracy:** 67.37%
- **PrivateTest loss:** 0.9236

## Model Analysis

The final V6 analysis includes random prediction inspection, a raw confusion matrix, per-class accuracy, a row-normalized confusion matrix, directed and bidirectional confusion-pair analysis, prediction-confidence analysis, and confidence-based selection of representative errors.

These analyses identify systematic error patterns before examples are selected for Grad-CAM; Grad-CAM cases are not chosen randomly.

## Grad-CAM

Grad-CAM is applied to V6's final convolutional layer. Gradients of a target pre-softmax class score are globally averaged to weight the layer's feature maps, producing a spatial heatmap for that class.

The implementation supports an explicitly selected target class. For a misclassified image, this allows separate comparison of Grad-CAM for the predicted class and the ground-truth class. Correctly classified examples are also used as controls when examining systematic confusion patterns. Detailed sample-level visualizations and interpretation remain in the notebook.

## Repository Structure

```text
.
├── data/                  # FER2013 CSV (not included)
├── notebooks/
│   └── explore.ipynb      # experiment comparison and final-model analysis
├── results/               # experiment configs, histories, and checkpoints
├── src/
│   ├── main.py            # datasets, models, training, and validation
│   ├── experiments.py     # experiment artifact utilities
│   ├── evaluate_final.py  # one-time V6 PrivateTest evaluation
│   └── gradcam.py         # Grad-CAM and visualization utilities
├── README.md
└── requirements.txt
```

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then download `fer2013.csv` and place it in `data/` as described above.

## Usage

Training runs are identified by a unique experiment name and save their artifacts under `results/<experiment>/`. For example, a V6-style run can be launched with:

```bash
python src/main.py \
  --experiment <new-experiment-name> \
  --model vgg_v2 \
  --epochs 40 \
  --augment \
  --early-stopping \
  --lr-scheduler
```

Evaluate the already-selected V6 checkpoint once on PrivateTest with:

```bash
python src/evaluate_final.py
```

Open `notebooks/explore.ipynb` for experiment comparison, final-model error analysis, and Grad-CAM visualizations.

## References

1. Y. Khaireddin and Z. Chen, “Facial Emotion Recognition: State of the Art Performance on FER2013,” *arXiv preprint arXiv:2105.03588*, 2021. [arXiv](https://arxiv.org/abs/2105.03588)
2. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” in *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017, pp. 618–626.

## Notes

- The FER2013 dataset is not included due to its size.
- PrivateTest is kept outside the normal development loop and is not used for training, validation, checkpoint selection, or hyperparameter tuning.

## License

This project is for educational and research purposes.
