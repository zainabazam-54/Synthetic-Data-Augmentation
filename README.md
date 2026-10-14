# GAN-Based Synthetic Data Augmentation for Pneumonia Classification

## Overview

This project explores the use of **Generative Adversarial Networks (GANs)** to generate synthetic chest X-ray images and address class imbalance in a pneumonia classification dataset. The goal is to increase the representation of the underrepresented `NORMAL` class and investigate how synthetic image augmentation affects CNN-based classification.

Implemented in Python using PyTorch, the project includes data analysis, image preprocessing, GAN training, synthetic image generation, CNN classification, and model evaluation.

## Objectives

- Analyze class imbalance in a chest X-ray dataset.
- Develop a GAN to generate synthetic images of normal lungs.
- Generate 1,000 synthetic chest X-ray images to supplement the training data.
- Train and evaluate a CNN for binary classification of `NORMAL` and `PNEUMONIA` images.
- Investigate classification errors using confusion matrices and class-specific evaluation metrics.
- Evaluate synthetic image quality using pixel statistics, Wasserstein distance, the Kolmogorov–Smirnov test, and Structural Similarity Index (SSIM).

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset by Paul Timothy Mooney, available on Kaggle.

**Dataset:** [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset contains chest X-ray images organized into two classes:

- `NORMAL` — images without pneumonia.
- `PNEUMONIA` — images showing pneumonia.

The notebook identifies an imbalance in the training set:

| Class | Number of images |
|---|---:|
| NORMAL | 1,341 |
| PNEUMONIA | 3,875 |

The GAN is trained using only the `NORMAL` training images to generate additional samples for the minority class.

## Project Workflow

### 1. Data Exploration and Preprocessing

- Analyzed class distributions across the training, validation, and test sets.
- Visualized representative images from both classes.
- Resized images to 64 × 64 pixels.
- Converted grayscale X-rays into three-channel images.
- Converted images to tensors and normalized pixel values for model training.

### 2. Baseline CNN Classifier

Implemented a custom CNN using PyTorch to classify chest X-ray images into `NORMAL` and `PNEUMONIA` categories.

The model includes convolutional layers, ReLU activations, max pooling, and fully connected layers. It is trained using cross-entropy loss and the Adam optimizer.

Baseline evaluation includes accuracy, precision, recall, F1-score, and a confusion matrix.

### 3. GAN Architecture and Training

Developed a GAN consisting of two neural networks:

**Generator**
- Takes a 100-dimensional random noise vector as input.
- Uses transposed convolutional layers, batch normalization, and ReLU activations.
- Generates three-channel images at 64 × 64 resolution.
- Uses a Tanh output activation.

**Discriminator**
- Uses convolutional layers to distinguish real chest X-rays from generated images.
- Incorporates batch normalization and LeakyReLU activations.
- Produces a score used to train the adversarial networks.

The generator and discriminator are trained adversarially for 30 epochs using binary cross-entropy with logits and Adam optimizers.

### 4. Synthetic Image Generation

After training, the generator produces **1,000 synthetic NORMAL chest X-ray images**.

These images are assigned the `NORMAL` class label and incorporated into the training pipeline to investigate their effect on classification.

### 5. Augmented CNN Training

The notebook explores retraining the CNN using synthetic images alongside real training images.

The later training pipeline combines the original training dataset with synthetic `NORMAL` samples and applies class-weighted cross-entropy loss to account for the remaining class imbalance.

### 6. Model Evaluation

The classifier is evaluated on the test dataset using:

- Accuracy
- Precision
- Recall (sensitivity)
- F1-score
- Confusion matrix
- Specificity

The notebook also examines misclassified images and compares real and synthetic image distributions to assess the quality of generated samples.

## Results

The notebook's reported evaluation after GAN augmentation and weighted-loss training is summarized below:

| Metric | Reported result |
|---|---:|
| Test accuracy | 80% |
| Weighted precision | 84% |
| Weighted recall | 80% |
| Weighted F1-score | 79% |
| PNEUMONIA recall (sensitivity) | 98.46% |
| NORMAL recall | 50.43% |

**Interpretation:** The classifier identifies most pneumonia cases in the reported evaluation, but its recall for the NORMAL class remains limited. This indicates that false-positive predictions remain an important issue and that further model tuning and validation are needed.

These figures are the results reported by the notebook, not a claim that GAN augmentation definitively improves performance over the baseline. A controlled comparison using the same held-out test set is needed to establish the actual impact of augmentation.

## Technologies Used

- **Language:** Python
- **Deep Learning:** PyTorch, Torchvision
- **Generative Modeling:** Generative Adversarial Networks (GANs)
- **Computer Vision:** Pillow, Matplotlib
- **Data Analysis:** Pandas, NumPy
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
-
