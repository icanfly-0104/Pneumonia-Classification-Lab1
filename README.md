# Pneumonia Classification Lab1

## Project Overview

This project aims to classify **chest X-ray images** into two categories:

- **NORMAL**
- **PNEUMONIA**

The project uses deep learning-based image classification to train a ResNet model that can identify whether a chest X-ray image shows signs of pneumonia.

---

## Method

The project uses **ResNet** as the image classification model.

The following ResNet architectures are supported:

- ResNet18
- ResNet50
- ResNet101

The model can use **ImageNet pretrained weights**.

Before training, chest X-ray images are resized to **224 × 224** and data augmentation is applied to the training images, including rotation, scaling, and contrast adjustment.

The model is trained using:

- **Adam Optimizer**
- **Weighted Cross Entropy Loss**

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Dataset

The project requires a chest X-ray dataset organized as:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The dataset contains two classes:

```text
NORMAL
PNEUMONIA
```

The `train` and `val` datasets are used for model training and validation, while the `test` dataset is used for final evaluation.

---

## Requirements

The project requires Python and the following packages:

```text
PyTorch
torchvision
NumPy
Matplotlib
Seaborn
Pillow
scikit-learn
tqdm
```

A CUDA-compatible GPU is recommended for training.

---

# How to Run

## 1. Train the Model

Run the training program with:

```bash
python train.py
```

The default configuration is:

```text
Model       : ResNet18
Pretrained  : Yes
Epochs      : 50
Batch Size  : 128
Learning Rate: 1e-4
Weight Decay: 1e-2
```

After training, the trained model is saved as:

```text
model_weights.pt
```

---

## 2. Test the Model

After training, run:

```bash
python test.py
```

The program loads `model_weights.pt` and evaluates the model using the test dataset.

---

# Command-line Arguments

## Training Arguments

The following arguments can be used with `train.py`:

| Argument | Default | Description |
|---|---:|---|
| `--model` | `resnet18` | Model architecture: `resnet18`, `resnet50`, or `resnet101` |
| `--weight` | `pretrain` | Use ImageNet pretrained weights or not (none) |
| `--num_classes` | `2` | Number of classification classes |
| `--num_epochs` | `50` | Number of training epochs |
| `--batch_size` | `128` | Training batch size |
| `--lr` | `1e-4` | Learning rate |
| `--wd` | `1e-2` | Weight decay |
| `--dataset` | `chest_xray` | Dataset directory |
| `--degree` | `10` | Maximum random rotation angle |
| `--resize` | `224` | Image size |
| `--scale` | `0.1` | Random scaling range |
| `--contrast` | `0.3` | Contrast augmentation |
| `--brightness` | `0.0` | Brightness augmentation |

### Example

Train a ResNet50 model with pretrained weights:

```bash
python train.py --model resnet50 --weight pretrain
```

Train a ResNet18 model with custom training parameters:

```bash
python train.py --model resnet18 --num_epochs 30 --batch_size 64 --lr 0.0001 --wd 0.01
```

Change the data augmentation settings:

```bash
python train.py --degree 10 --scale 0.1 --contrast 0.3 --brightness 0.1
```

---

## Testing Arguments

The following arguments can be used with `test.py`:

| Argument | Default | Description |
|---|---|---|
| `--model` | `resnet18` | Model architecture: `resnet18`, `resnet50`, or `resnet101` |
| `--weight` | `pretrain` | Model initialization setting |
| `--path` | `model_weights.pt` | Path to the trained model weights |

### Example

Test a ResNet18 model:

```bash
python test.py --model resnet18 --path model_weights.pt
```

Test a ResNet50 model:

```bash
python test.py --model resnet50 --path model_weights.pt
```

Use a different model weight file:

```bash
python test.py --model resnet18 --path my_model.pt
```

> **Note:** The model architecture specified during testing must match the architecture used to generate the trained weight file.

---

# Output

After training and testing, the project produces the following outputs.

### Model

```text
model_weights.pt
```

Contains the trained model parameters.

### Training Results

```text
accuracy_curve.png
loss_curve.png
validation_metrics.png
```

These plots show the training and validation performance.

### Test Results

The testing program reports:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

It also generates:

```text
confusion_matrix.png
```

which shows the classification results for `NORMAL` and `PNEUMONIA`.

---

## Project Goal

The goal of this project is to develop and evaluate a deep learning model capable of automatically classifying chest X-ray images as **NORMAL** or **PNEUMONIA**, and to analyze its performance using standard classification metrics and a confusion matrix.