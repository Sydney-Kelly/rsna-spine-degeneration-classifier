# Lumbar Spine Degenerative Classification
### Deep Learning for Medical Image Analysis | RSNA 2024 Kaggle Competition

This project applies convolutional neural networks to classify the severity of degenerative spine conditions from lumbar MRI scans. Using the RSNA 2024 Lumbar Spine Degenerative Classification dataset, we trained and compared a custom CNN and a fine-tuned VGG16 model to simultaneously predict spinal condition type and severity — with the goal of supporting early detection of lumbar spine degeneration.

---

## Results

| Model | Condition Accuracy | Severity Accuracy |
|---|---|---|
| Custom CNN | ~79% | ~65% |
| Fine-tuned VGG16 | **95%** | **79%** |

The fine-tuned VGG16 model significantly outperformed the custom CNN on both tasks, demonstrating the value of transfer learning on a domain-specific medical imaging problem.

---

## Problem Statement

Lumbar spine degeneration is a leading cause of chronic pain and disability worldwide. Early and accurate classification of degenerative conditions — such as Spinal Canal Stenosis, Neural Foraminal Narrowing, and Subarticular Stenosis — can support clinical decision-making and improve patient outcomes.

This project frames the problem as a **multi-label classification task**, predicting:
- **Condition** (5 classes): Spinal Canal Stenosis, Left/Right Neural Foraminal Narrowing, Left/Right Subarticular Stenosis
- **Severity** (3 classes): Normal/Mild, Moderate, Severe

---

## Dataset

This project uses the [RSNA 2024 Lumbar Spine Degenerative Classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification) dataset from Kaggle.

- ~32GB of DICOM format MRI images
- ~1,900 patient studies
- Three MRI series types: Sagittal T1, Sagittal T2/STIR, Axial T2
- Labeled coordinates for 5 spinal conditions across 5 vertebral levels (L1/L2 through L5/S1)

> **Note:** You must create a Kaggle account and accept the competition rules before downloading the data. The notebook handles the download automatically via the Kaggle API.

---

## Models

### Model 1: Custom CNN
A 3-layer convolutional network built from scratch with dual output heads for simultaneous condition and severity prediction. Uses an exponential learning rate schedule and weighted loss functions to address class imbalance.

### Model 2: Fine-tuned VGG16
A pre-trained VGG16 model (ImageNet weights) with the last 5 layers unfrozen for domain adaptation. Features separate dense branches for condition and severity prediction, with spatial metadata (x/y coordinates, vertebral level) concatenated to the shared feature representation.

---

## Repo Structure

```
rsna-spine-degeneration-classifier/
├── README.md
└── notebooks/
    └── rsna_spine_classification.ipynb
```

---

## Getting Started

### Prerequisites
- Google Colab (recommended) or local Python environment
- Kaggle account with competition rules accepted
- GPU runtime strongly recommended (T4 or better)

### Setup

1. **Clone the repo**
```bash
git clone https://github.com/Sydney-Kelly/rsna-spine-degeneration-classifier.git
```

2. **Add Kaggle credentials to Colab Secrets**

In Google Colab, go to **Tools > Secrets** and add:
- `KAGGLE_USERNAME` — your Kaggle username
- `KAGGLE_KEY` — your Kaggle API key

3. **Open the notebook in Colab and run all cells**

The notebook will automatically download and unzip the dataset, preprocess the images, train both models, and output evaluation metrics.

---

## Key Libraries

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| pydicom | Reading DICOM medical images |
| Albumentations | Data augmentation |
| OpenCV | Image resizing and preprocessing |
| scikit-learn | Train/val/test splitting, evaluation metrics |
| pandas / numpy | Data manipulation |
| seaborn / matplotlib | Visualization |

---

## Acknowledgements

Dataset provided by the Radiological Society of North America (RSNA) via Kaggle. This project was completed as part of the DSBA 6165: Artificial Intelligence and Deep Learning course at UNC Charlotte.
