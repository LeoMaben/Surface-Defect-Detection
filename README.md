# Surface Defect Detection using CNNs

This project explores and compares deep learning techniques for **binary classification of defective vs. non-defective 
materials** in grayscale surface images. It aims to deepen my understanding of CNN architectures, model performance
evaluation, and explainability using tools like **Grad-CAM**.

---

## Project Objectives

- Build a robust image classification pipeline for defect detection
- Explore and compare different CNN architectures
- Apply data augmentation and preprocessing best practices
- Understand model behavior through Grad-CAM explainability
- Automate model evaluation using key metrics
- Package project in a clean, modular way for reproducibility and future expansion

---

## Roadmap

### Phase 1: Data & Preprocessing
- [x] Load and resize images from source folders
- [x] Assign binary labels (defect / ok)
- [x] Perform stratified train-test split
- [x] Apply data augmentation (horizontal flip, rotation, contrast, etc.)
- [x] Normalize and batch the data using a `tf.data.Dataset` pipeline

### Phase 2: Modeling
- [x] Build multiple CNN models from scratch:
  - **LeNet-inspired model** (2 conv layers)
  - **Custom ResNet architecture**
  - **AlexNet architecture**
- [x] Train each model
- [x] Apply EarlyStopping and regularization (Dropout, L2, BatchNorm)

### Phase 3: Evaluation
- [x] Evaluate test set accuracy, loss
- [x] Generate:
  - Confusion matrix
  - Precision, recall, F1-score
  - ROC curve
- [x] Automate comparisons across models

### Phase 4: Explainability
- [x] Use **Grad-CAM** to visualize decision-making of CNNs
- [x] Overlay heatmaps on sample defect images for inspection
---

## Models & Accuracy (Test Set)

| Model                             | Accuracy | Notes |
|-----------------------------------|----------|-------|
| **LeNet without Augmentation**    | ~x%      | L2, EarlyStopping |
| **LeNet with Augmentation**       | ~x%      | L2, EarlyStopping |
| **ResNet without Augmentations**  | ~x%      | Custom-built lightweight ResNet |
| **ResNet with Augmentations**     | ~x%      | Custom-built lightweight ResNet |
| **AlexNet without Augmentations** | ~x%      | Struggled to converge, possible overfitting |
| **AlexNet with Augmentations**    | ~x%      | Struggled to converge, possible overfitting |

---

## 🧰 Tech Stack

- Python 3.9
- TensorFlow / Keras
- NumPy, OpenCV
- Matplotlib, Scikit-learn
- Albumentations (for augmentation)
- Grad-CAM visualization

---
