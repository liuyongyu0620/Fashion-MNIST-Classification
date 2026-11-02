## Project Overview

This project reconstructs a multilayer perceptron (MLP) using Python and NumPy to classify four Fashion-MNIST clothing categories: T-shirt/top, Pullover, Coat, and Shirt.

The main objective is to understand the fundamentals of neural network training through manual implementation of forward propagation, backpropagation, and gradient descent without relying on deep learning frameworks.

## Technologies

- Python
- NumPy
- Matplotlib

## Model Architecture

- Input layer: 784 features (28 × 28 grayscale images)
- Hidden layer: 128 neurons
- Output layer: 4 neurons with Softmax
- Loss function: Categorical Cross-Entropy
- Optimization: Mini-batch Gradient Descent
- Activation functions compared: ReLU and Sigmoid

## Experimental Results

| Metric | ReLU | Sigmoid |
|---|---|---|
| Training Accuracy | 84.69% | 76.29% |
| Validation Accuracy | 80.25% | 76.25% |
| Validation Loss | 0.5061 | 0.6063 |

Under the tested configuration, ReLU achieved higher validation accuracy than Sigmoid.

## Training Curves

### ReLU

![ReLU Training Curves](results/curves_relu.png)

### Sigmoid

![Sigmoid Training Curves](results/curves_sigmoid.png)
