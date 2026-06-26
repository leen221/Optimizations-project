# Neural Network Classifier for IRIS Dataset
## Term Project Report

---

## 1. Introduction

This project implements a Neural Network classifier for the IRIS dataset, comparing three training configurations:
1. **No Regularization** — baseline model
2. **LASSO (L1 Regularization)** — promotes sparsity in weights
3. **Ridge (L2 Regularization)** — penalizes large weights

The error metric used is **Mean Squared Error (MSE)**, discussed in the context of Chapters 6 and 7.

---

## 2. Dataset Description

| Property | Value |
|----------|-------|
| Training samples | 120 |
| Test samples | 30 |
| Input features | 4 (sepal/petal length & width) |
| Output classes | 3 (Setosa=0, Versicolor=1, Virginica=2) |

---

## 3. Network Architecture

```
Input Layer (4 neurons)
        ↓
Hidden Layer 1 (16 neurons, ReLU)
        ↓
Hidden Layer 2 (8 neurons, ReLU)
        ↓
Output Layer (3 neurons, Softmax)
```

- **Optimizer**: Mini-batch Gradient Descent (batch=16, lr=0.05)
- **Epochs**: 1000
- **Weight Init**: Xavier initialization

---

## 4. Error Estimation — MSE (Chapter 6 & 7)

### Chapter 6 — Approximation and Fitting
The neural network is essentially a function approximator. We seek weights **W** that minimize the fitting error between predicted output **ŷ** and true output **y**:

> **MSE = (1/n) · Σ (ŷᵢ - yᵢ)²**

This is the **least-squares approximation** problem from Chapter 6. The network approximates the true classification function f(x) by minimizing the sum of squared residuals over the training set.

### Chapter 7 — Statistical Estimation
From a statistical perspective (Chapter 7), MSE is decomposed as:

> **MSE = Bias² + Variance + Irreducible Noise**

- **Bias**: How far on average the model predictions are from the truth
- **Variance**: How much the model predictions fluctuate across datasets
- **Regularization** directly controls this trade-off (the Bias-Variance Tradeoff)

---

## 5. Regularization Methods

### 5.1 No Regularization
The loss function is pure MSE:

> **L = MSE = (1/n) · Σ (ŷᵢ - yᵢ)²**

The model is free to use any weight values. Risk of **overfitting** when training data is limited.

### 5.2 LASSO — L1 Regularization
Adds the sum of absolute weight values as a penalty:

> **L = MSE + λ · Σ|wⱼ|**

- **Effect**: Drives some weights exactly to **zero** → feature selection
- **Gradient update**: `dW += λ · sign(W)`
- **Chapter 6 connection**: This is the **L1-norm** approximation from the Fitting chapter

### 5.3 Ridge — L2 Regularization
Adds the sum of squared weight values as a penalty:

> **L = MSE + λ · Σwⱼ²**

- **Effect**: Shrinks all weights toward zero but **never exactly zero**
- **Gradient update**: `dW += 2λ · W`
- **Chapter 6 connection**: This is the **L2-norm (Tikhonov regularization)** from the Fitting chapter

---

## 6. Results

### 6.1 Accuracy

| Model | Train Accuracy | Test Accuracy | Test MSE |
|-------|---------------|---------------|----------|
| No Regularization | 100.00% | 96.67% | 0.022076 |
| LASSO (L1) | 100.00% | 96.67% | 0.021972 |
| Ridge (L2) | 100.00% | 96.67% | 0.021523 |

### 6.2 Confusion Matrix (All Models)

```
              Setosa  Versicolor  Virginica
Setosa           8         0         0
Versicolor       0        14         0
Virginica        0         1         7
```

One Virginica sample was misclassified as Versicolor. This is expected, as Versicolor and Virginica have overlapping features in the IRIS dataset.

---

## 7. Discussion and Justification

### Why do all three models achieve similar accuracy?
The IRIS dataset is a small, clean, linearly-separable (mostly) dataset. With only 120 training samples, the model does not have enough complexity to overfit severely. Therefore:
- The **No Regularization** model already generalizes well
- **LASSO** and **Ridge** provide marginal improvement in MSE (lower values) by constraining the weight space
- The improvement in MSE (0.022 → 0.0215) confirms that regularization slightly reduces variance

### Statistical Interpretation (Chapter 7)
From Chapter 7 — **Maximum Likelihood Estimation (MLE)**:
- No Regularization corresponds to **pure MLE**: maximize likelihood of observed data
- L2 (Ridge) corresponds to **Maximum A Posteriori (MAP)** estimation with a **Gaussian prior** on weights
- L1 (LASSO) corresponds to **MAP** estimation with a **Laplacian prior** on weights

### Approximation Interpretation (Chapter 6)
From Chapter 6 — **Least Squares Fitting**:
- All three models solve a **least squares problem** (MSE = squared error)
- Ridge adds an **L2 regularization term** → equivalent to Tikhonov regularization in linear algebra
- LASSO adds an **L1 regularization term** → equivalent to sparse recovery / basis pursuit

---

## 8. Execution Steps

```bash
# Step 1: Install requirements
pip install numpy pandas matplotlib

# Step 2: Place data files in same folder
# iris_training.csv
# iris_test.csv

# Step 3: Run the classifier
python iris_classifier.py

# Step 4: View outputs
# - Console: accuracy table + confusion matrices
# - File: iris_results.png (visualization dashboard)
```

---

## 9. Conclusion

The Neural Network classifier successfully classifies IRIS flowers with **96.67% test accuracy** across all three configurations. Ridge (L2) regularization achieved the **lowest MSE (0.0215)**, confirming that it reduces prediction variance the most on this dataset. LASSO (L1) provided marginal improvement over no regularization, as the dataset does not have many irrelevant features to zero out.

From the theoretical perspective (Chapters 6 & 7), MSE minimization is equivalent to least-squares approximation (Ch.6) and maximum likelihood estimation (Ch.7), while regularization provides a Bayesian prior that controls the bias-variance tradeoff.

---

*Project submitted for term evaluation. Presentation scheduled: May 7, 2026.*
