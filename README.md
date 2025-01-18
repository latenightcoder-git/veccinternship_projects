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
[Check the code in Jupyter Notebook]("C:\veccinternship_projects\XOR_Backpropagation_md.ipynb")

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


