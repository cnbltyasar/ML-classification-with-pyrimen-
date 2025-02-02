# ML-classification-with-pyrimen-
# EEG Classification with PyRiemann

This repository contains code for EEG signal classification using covariance matrices, Tangent Space mapping, and various machine learning classifiers. The project uses [PyRiemann](https://pyriemann.readthedocs.io/) and [MNE](https://mne.tools/) libraries to process EEG data and evaluate multiple classifiers.

## Features

- **Preprocessing:** Band-pass filtering (2–30 Hz) of EEG signals.
- **Covariance Estimation:** Compute covariance matrices per epoch.
- **Re-centering:** Use parallel transport to align covariance matrices.
- **Tangent Space Projection:** Map covariance matrices to a Euclidean space.
- **Classification:** Evaluate models including:
  - Linear Discriminant Analysis (LDA)
  - Logistic Regression (LR)
  - Support Vector Machine (SVC)
  - Lasso Regression
  - Minimum Distance to Mean (MDM)
  - Decision Tree Classifier
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cnbltyasar/ML-classification-with-pyrimen-
   cd your-repo
