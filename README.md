#  Student Performance Prediction using Neural Network

A simple **Neural Network built from scratch using Python and NumPy** to predict whether a student is likely to **Pass or Fail** based on study hours, attendance, previous marks, and assignment scores.

This project demonstrates fundamental concepts of **Artificial Neural Networks, Forward Propagation, ReLU, Sigmoid, Binary Cross-Entropy, Backpropagation, and Gradient Descent** without using TensorFlow or PyTorch.

---

##  Project Overview

The model takes four student performance features as input:

- Study Hours
- Attendance Percentage
- Previous Marks
- Assignment Scores

The neural network then predicts:

```text
0 → Fail
1 → Pass
```

## Requirements

```
pip install numpy matplotlib
```

## How the Neural Network Works

The neural network works by passing the student data through several mathematical steps:

```text
Input Data
    ↓
Feature Scaling
    ↓
Weighted Sum
    ↓
ReLU Activation
    ↓
Weighted Sum
    ↓
Sigmoid Activation
    ↓
Prediction Probability
    ↓
Pass / Fail
```
## Core Formula

Feature Scaling:
```
X_scaled = (X - μ) / σ
```

First Layer:
```
Z₁ = XW₁ + b₁
```

ReLU:
```
A₁ = max(0, Z₁)
```

Second Layer:
```
Z₂ = A₁W₂ + b₂
```

Sigmoid:
```
A₂ = 1 / (1 + e^(-Z₂))
```

Binary Cross-Entropy:
```
Loss = -(1/m)Σ[Y log(A₂) + (1-Y)log(1-A₂)]
```

Output Gradient:
```
dZ₂ = A₂ - Y
```

Weight Update:
```
W_new = W_old - α × dW
```

##  Conclusion

This project demonstrates how a simple neural network can predict student **Pass/Fail** results using study hours, attendance, previous marks, and assignment scores. By implementing **ReLU, Sigmoid, forward propagation, backpropagation, and gradient descent from scratch**, the project provides a clear understanding of how neural networks learn and make predictions.
