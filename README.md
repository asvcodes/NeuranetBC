
# Neural Networks: Theory

![a Neural Network - DAG](https://d3lkc3n5th01x7.cloudfront.net/wp-content/uploads/2023/05/30234805/What-are-neural-networks-Banner.svg "a Neural Network - DAG")

A neural network is a method in artificial intelligence that teaches computers to process data in a way that is inspired by the human brain. It is a type of machine learning process, called deep learning, that uses interconnected nodes or neurons in a layered structure that resembles the human brain.

## 1. Neurons: The Building Blocks

Artificial neurons are the fundamental units of neural networks, inspired by biological neurons in the brain. They process and transmit information through the network.

### Key Components:
- **Inputs**: Receive data from other neurons or external sources.
- **Weights**: Determine the importance of each input.
- **Bias**: Adds a constant term to adjust the neuron's activation.
- **Activation Function**: Introduces non-linearity, allowing the network to learn complex patterns.

### Mathematical Representation:
output = activation_function( ∑ (inputs × weights) + bias)

## 2. Layers: Structuring the Network

Neural networks are organized into layers of neurons, each serving a specific purpose in the information processing pipeline.

### Types of Layers:
1. **Input Layer**: Receives raw data.
2. **Hidden Layers**: Process information and extract features.
3. **Output Layer**: Produces the final prediction or classification.

## 3. Multi-Layer Perceptron (MLP)

An MLP is a type of feedforward neural network consisting of multiple layers of neurons. It's capable of learning complex, non-linear relationships in data.

### Architecture:
- Input Layer → Hidden Layer(s) → Output Layer

### Key Features:
- Fully connected layers.
- Non-linear activation functions (e.g., ReLU, Sigmoid, Tanh).
- Capable of universal function approximation.

## 4. Backpropagation and Gradient Descent

Backpropagation and gradient descent are the core algorithms used to train neural networks.

### Backpropagation:
A method to calculate gradients of the loss function with respect to the network's parameters (weights and biases).

### Gradient Descent:
An optimization algorithm that iteratively adjusts the network's parameters to minimize the loss function.

## 5. Training Process: Forward Pass, Backward Pass, and Optimization

### Forward Pass:
1. Input data flows through the network.
2. Each neuron computes its output.
3. The final layer produces the prediction.

### Backward Pass (Backpropagation):
1. Calculate the error (difference between prediction and actual output).
2. Compute gradients of the loss with respect to each parameter.
3. Propagate the error backwards through the network.

### Optimization:
1. Update weights and biases using gradients and learning rate.
2. Repeat the process for multiple epochs.

### Mathematical Representation:
Parameter Update:
𝜃new = 𝜃old − learning_rate × gradient


## 6. Key Concepts in Training

- **Loss Function**: Measures the difference between predicted and actual outputs.
- **Learning Rate**: Controls the step size in parameter updates.
- **Epochs**: Number of times the entire dataset is passed through the network.
- **Batch Size**: Number of samples processed before parameter update.

## 7. Advanced Techniques

- **Regularization**: Techniques like L1, L2, and Dropout to prevent overfitting.
- **Optimization Algorithms**: Advanced methods such as Adam and RMSprop.
- **Batch Normalization**: Normalizes inputs of each layer to improve training.
- **Transfer Learning**: Leveraging pre-trained models for new tasks.

# Repository Overview

## Main

Main has a compact Autograd engine that implements backpropagation (reverse-mode autodiff) over a dynamically built Directed Acyclic Graph (DAG) along with a very small neural net lib. The DAG operates exclusively on scalar values, which means each neuron is broken down into individual tiny addition and multiplication operations. Despite this granular approach, it is capable of constructing entire deep neural networks for binary classification, as demonstrated in the accompanying notebook.

### Installation

To install Main, use the following command:

```bash
pip install Main
```

### Training a Neural Network

The `trail.ipynb` notebook provides a comprehensive demonstration of training a 2-layer neural network (MLP) for binary classification. This involves initializing a neural network from the `Main.neuralnet` module, implementing a simple Support Vector Machine (SVM) "max-margin" binary classification loss, and optimizing using Stochastic Gradient Descent (SGD). As illustrated in the notebook, employing a 2-layer neural network with two 16-node hidden layers results in an effective decision boundary on the moon dataset.

### License

This repository is licensed under the MIT License. This permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.

---