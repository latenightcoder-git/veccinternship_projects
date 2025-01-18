********************************************************************************************************************************
# Year - 2024
# VECC Internship Projects
Welcome to the VECC Internship Projects repository. This collection showcases the work completed by me during the Variable Energy Cyclotron Centre (VECC) CSE internship, focusing on neural network implementations and activation functions. I learnt quite a lot about deep learning under the guidance of my guide Shri Monirul Purkait, and completed my summer training on "Deep Learning for Computer vision with Python" .

Repository Contents
The repository includes the following Jupyter Notebooks:

XOR_Backpropagation_md.ipynb: Implementation of the XOR logic gate using backpropagation.
XORdeeplearning_md.ipynb: Deep learning approach to solving the XOR problem.
md_sigmoid_neuron.ipynb: Exploration of the sigmoid neuron model.
relu_md.ipynb: Analysis and implementation of the ReLU activation function.
********************************************************************************************************************************

## 1. XOR Problem with Backpropagation 
[Check the code in Jupyter Notebook](XOR_Backpropagation_md.ipynb)

### XOR Problem and Neural Network Implementation

#### **XOR Logic Gate**
The XOR (exclusive OR) gate outputs `1` only when the inputs are different (e.g., `0 XOR 1 = 1`). It’s a non-linearly separable problem, meaning it cannot be solved using a simple linear model, such as a perceptron.

#### **Neural Network Architecture**
The neural network used to solve the XOR problem typically has:
- **Input Layer**: Takes the two binary inputs (e.g., 0, 1).
- **Hidden Layer**: A layer of neurons with activation functions, providing the network with non-linear capabilities.
- **Output Layer**: Outputs a binary result (either 0 or 1).

#### **Backpropagation Algorithm**
1. **Forward Propagation**:
   - Inputs pass through the network layers, with weights applied to each connection.
   - The output is computed from the hidden layer to the output layer.

2. **Loss Function**:
   - The difference between the predicted and actual outputs is calculated. Commonly used is the Mean Squared Error (MSE) or Cross-Entropy loss.

3. **Backward Propagation**:
   - The error (loss) is propagated backward through the network.
   - Gradients of the loss with respect to each weight are calculated using the chain rule.
   - The weights are updated using optimization methods (e.g., gradient descent) to minimize the loss.

#### **Key Concepts in Backpropagation**
- **Gradient Descent**: An optimization method to minimize the loss function by adjusting weights.
- **Activation Function**: A mathematical function like the sigmoid or ReLU, applied at each neuron, to introduce non-linearity into the network.
- **Learning Rate**: A parameter that controls how much the weights are adjusted during each update.

This neural network training process helps the model learn the XOR pattern through iterative updates, gradually reducing the error in predictions.

## XOR deeplearning
[Check the code in Jupyter Notebook](XORdeeplearning_md.ipynb)

**Introduction**: Delves into deep learning techniques for solving the XOR problem.

**Deep Neural Network**: 
- A "deep" learning approach uses multiple layers in the neural network (hidden layers) to learn more complex patterns. Implements a deeper neural network, possibly with more hidden layers, compared to the previous notebook.
- Uses advanced neural network architectures to improve performance.

**Training and Optimization**:
- Explains the training process for deep neural networks, including optimization techniques like gradient descent.
- May include regularization techniques to prevent overfitting.

**Architecture**:
Input layer → Hidden layer(s) → Output layer.

**Activation Functions**:
Used in neurons to introduce non-linearity, helping the model solve complex problems like XOR.

**Key Concepts**:
* Deep Learning: Involves neural networks with multiple hidden layers.
* Optimization: Techniques to adjust weights for minimizing the loss function.
* Regularization: Methods to reduce overfitting and improve generalization.

##  Sigmoid Neuron Model 
[Check the code in Jupyter Notebook](sigmoid_neuron.ipynb)

**What is a Sigmoid Neuron?**
- A basic unit of neural networks where the activation function is sigmoid:
<img src="images\sigmoid.png" alt="Sigmoid formula" title="Sigmoid formula pic">
It squashes the input into a range between 0 and 1.

**Sigmoid Activation Function** : <img src="images\sigmoid.png" alt="Sigmoid formula" title="Sigmoid formula pic">
- Explains how the sigmoid function outputs values between 0 and 1, making it suitable for binary classification.

**Implementation**: 
    - Code implementation of a sigmoid neuron, including forward and backward propagation steps.
    - Calculates the gradient of the sigmoid function for use in backpropagation.
    
    Why Use It?
        Useful for binary classification and probabilistic outputs.
    
**Limitations**:
Vanishing gradient problem: Gradients become too small during backpropagation for large networks, slowing learning.

**Key Concepts**:
Sigmoid Function: A smooth, differentiable function that outputs values between 0 and 1.

**Gradient**: The derivative of the sigmoid function, used for updating weights during training.

## ReLU Activation Function
[Check the code in Jupyter Notebook](relu_md.ipynb)

**What is ReLU**?
- Rectified Linear Unit (ReLU) is a popular activation function: f(x)=max(0,x)
- RelU Activation function described the RelU function, explains how RelU introduces non-linearity into the model, allowing the network to learn complex patterns.

**Why Use ReLU**?
- Computationally efficient.
- Helps with the vanishing gradient problem by not saturating for positive values.

**Variants**:
Leaky ReLU, Parametric ReLU, etc., are used to address issues like "dead neurons."

**Implementation**:
- Code implementation of a ReLU neuron, including forward and backward propagation steps.
- Discusses the advantages and drawbacks of ReLU, such as the issue of dying ReLUs.

**Key Concepts**:
- ReLU Function: Outputs the input if positive, otherwise zero.
- Non-Linearity: Introduced by ReLU, enabling the learning of complex patterns.
- Dying ReLUs: A problem where neurons stop activating, often addressed with variants like Leaky ReLU.