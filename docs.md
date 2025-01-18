# Theory

## Introduction to Deep Learning
Deep learning is a subset of machine learning that focuses on algorithms inspired by the structure and function of the brain called artificial neural networks. It is widely used for tasks like image recognition, speech processing, and natural language understanding.

### Key Concepts in Deep Learning

1. Neural Networks
Structure:
Input Layer: Accepts data (e.g., images, text).
Hidden Layers: Perform computations and extract features.
Output Layer: Provides predictions or classifications.
Neuron (Node):
Takes weighted inputs, applies an activation function, and produces an output.

2. Forward Propagation
Data flows through the network:
Input → Hidden Layers → Output.
Each neuron computes a weighted sum of inputs:
<img src="images\zformula.png" alt="Zformula" title="Each neuron computes a weighted sum of inputs">
where 𝑤 are weights, 𝑥 are inputs, and 𝑏 is the bias term.

3. Activation Functions
Introduce non-linearity, enabling the network to learn complex patterns.
Common types:
<img src="images\zallformulas.png" alt="All formulas" title="All formulas">

4. Loss Function
Measures how far predictions are from the actual values.
Examples:
- Mean Squared Error (MSE): For regression tasks.
- Cross-Entropy Loss: For classification tasks.

5. Backpropagation
Algorithm for training neural networks.
Steps:
- Compute the loss.
- Calculate gradients of the loss with respect to weights using the chain rule.
- Update weights using gradients.

6. Gradient Descent
Optimization algorithm to minimize the loss function.
Variants:
- Stochastic Gradient Descent (SGD): Updates weights for each training example.
- Mini-Batch Gradient Descent: Uses small batches of training data.
- Adam: Combines momentum and adaptive learning rates.

### Key Terms

1. Epoch: One complete pass through the entire training dataset.
2. Batch Size: Number of training examples used in one forward and backward pass.
3. Learning Rate: Controls how much to adjust weights during training.
4. Overfitting: When the model performs well on training data but poorly on unseen data.
5. Solution: Use regularization, dropout, or more data.
6. Underfitting: When the model cannot capture patterns in the data due to insufficient complexity.
7. Model Parameters: Weights and biases learned by the model.
8. Hyperparameters: Configurable settings like learning rate, number of layers, etc.

### How Deep Learning Works
- Data is passed through the input layer.
- Hidden layers extract features by performing matrix multiplications, applying activation functions, etc.
- The output layer makes predictions based on extracted features.
- The loss function evaluates the predictions.
- Backpropagation adjusts the weights to minimize the loss.

### Common Deep Learning Architectures
1. Feedforward Neural Networks (FNN): 
    - Data flows in one direction. 
    - Used for basic tasks like regression and classification.
2. Convolutional Neural Networks (CNN): 
    - Specialized for image and video processing. 
    - Uses convolutional layers to detect spatial features.
3. Recurrent Neural Networks (RNN): 
    - Processes sequential data (e.g., text, time series). 
    - Maintains information through hidden states.
4. Transformers: 
    - Foundation for modern NLP models like BERT and GPT. 
    - Focuses on relationships between words in a sequence using self-attention.

### Deep Learning Frameworks
1. TensorFlow:
    - Developed by Google, supports large-scale machine learning.
2. PyTorch:
    - Popular for research, flexible and dynamic computation graph.
3. Keras:
    - High-level API for TensorFlow, user-friendly.

### Steps to Build a Deep Learning Model
1. Collect Data: Gather labeled data.
2. Preprocess Data: Normalize, augment, or clean the data.
3. Build the Model: Define layers, activation functions, and loss function.
4. Train the Model: Use training data to adjust weights via backpropagation.
5. Evaluate the Model: Test performance on unseen data.
6. Fine-tune Hyperparameters: Adjust learning rate, number of layers, etc.
