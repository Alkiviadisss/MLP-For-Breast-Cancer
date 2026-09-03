# PyTorch MLP for Breast Cancer Classification

This repository contains a streamlined, PyTorch implementation of a Multi-Layer Perceptron (MLP) built to classify data from the Breast Cancer Wisconsin dataset.

It serves as a foundational boilerplate for handling tabular data in PyTorch using neural network architecture, demonstrating best practices such as data splitting, proper feature scaling, and tracking validation metrics during the training loop.

## Results
The model converges rapidly and achieves high accuracy, demonstrating that a MLP is highly effective for this dataset.

<img width="1200" height="500" alt="Image" src="https://github.com/user-attachments/assets/3a8c44ed-845c-46c3-9450-f16f4f207cb8" />

**Output:**
```text
X shape: (569, 30)
y shape: (569,)
Train: (455, 30)
Validation: (57, 30)
Test: (57, 30)
MLP(
  (network): Sequential(
    (0): Linear(in_features=30, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=2, bias=True)
  )
)
Epoch [10/100] Train Loss: 0.5361 Val Loss: 0.5226 Val Accuracy: 0.8947
Epoch [20/100] Train Loss: 0.3795 Val Loss: 0.3705 Val Accuracy: 0.9825
Epoch [30/100] Train Loss: 0.2420 Val Loss: 0.2296 Val Accuracy: 1.0000
Epoch [40/100] Train Loss: 0.1539 Val Loss: 0.1308 Val Accuracy: 1.0000
Epoch [50/100] Train Loss: 0.1057 Val Loss: 0.0790 Val Accuracy: 1.0000
Epoch [60/100] Train Loss: 0.0801 Val Loss: 0.0551 Val Accuracy: 1.0000
Epoch [70/100] Train Loss: 0.0663 Val Loss: 0.0421 Val Accuracy: 1.0000
Epoch [80/100] Train Loss: 0.0574 Val Loss: 0.0341 Val Accuracy: 1.0000
Epoch [90/100] Train Loss: 0.0509 Val Loss: 0.0298 Val Accuracy: 1.0000
Epoch [100/100] Train Loss: 0.0458 Val Loss: 0.0277 Val Accuracy: 1.0000

Final Test Accuracy: 0.9298245906829834
```

## Dataset
This project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**.
* **Features:** 30 numerical features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
* **Target:** Binary classification (Malignant or Benign).

## Prerequisites
To run this code, you will need the following libraries installed:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Model Architecture
The neural network is a Feed-Forward MLP built using `torch.nn.Sequential`:
* **Input Layer:** 30 features
* **Hidden Layer 1:** 64 units, ReLU activation
* **Hidden Layer 2:** 32 units, ReLU activation
* **Output Layer:** 2 units (Binary Classification)

## Execution Pipeline
The script (`MLP.py`) follows a strict machine learning pipeline:
1. **Data Loading:** Loads the dataset using Pandas.
2. **Stratified Splitting:** Splits data into 80% Train, 10% Validation, and 10% Test.
3. **Scaling:** Uses `StandardScaler` (fit *only* on training data to prevent data leakage).
4. **Tensor Conversion:** Converts NumPy arrays to PyTorch Tensors.
5. **Training Loop:** Trains using the Adam optimizer and CrossEntropyLoss over 100 epochs.
6. **Validation:** Evaluates accuracy and loss on the validation set at the end of each epoch.
7. **Testing:** Runs a final evaluation on the unseen test set.
