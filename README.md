# MNIST Handwritten Digit Recognizer

A handwritten digit recognition project built **from scratch using Python and NumPy** to understand the fundamentals of neural networks, forward propagation, backpropagation, and gradient descent.

The model is trained on the **MNIST handwritten digit dataset** and can classify images of handwritten digits from **0 to 9**.

---

## Project Overview

The goal of this project was not just to use a machine learning library to train a model, but to understand what happens **inside a neural network**.

The neural network was implemented manually using NumPy, including:

- Parameter initialization
- Forward propagation
- ReLU activation
- Softmax activation
- One-hot encoding
- Backpropagation
- Gradient calculation
- Gradient descent
- Parameter updates
- Prediction and accuracy evaluation

The trained model achieved **90%+ training accuracy** with a relatively small neural network.

---

## Model Architecture

The network uses a simple fully connected architecture:

```text
28 × 28 Image
     │
     ▼
784 Input Features
     │
     ▼
┌─────────────────┐
│ Hidden Layer    │
│ 10 Neurons      │
│ ReLU            │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Layer    │
│ 10 Neurons      │
│ Softmax         │
└────────┬────────┘
         │
         ▼
   Digit Prediction
      0 - 9
```

### Layer dimensions

```text
Input:   784 × m
W1:      10 × 784
b1:      10 × 1

Hidden:  10 × m

W2:      10 × 10
b2:      10 × 1

Output:  10 × m
```

Where `m` represents the number of training examples.

---

## Dataset

The project uses the **MNIST handwritten digit dataset**.

Each image:

```text
28 × 28 pixels
```

is flattened into:

```text
784 features
```

Each pixel originally contains a value between:

```text
0 - 255
```

The pixel values are normalized to:

```text
0 - 1
```

using:

```python
X_train = X_train / 255.
X_dev = X_dev / 255.
```

This normalization significantly improves the stability of neural-network training.

---

## Neural Network Implementation

### 1. Parameter Initialization

The weights and biases are initialized using NumPy:

```python
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return W1, b1, W2, b2
```

### 2. ReLU Activation

```python
def ReLU(Z):
    return np.maximum(Z, 0)
```

ReLU introduces non-linearity into the network.

### 3. Softmax

The output layer uses Softmax to convert the output values into probabilities:

```python
def softmax(Z):
    A = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return A / np.sum(A, axis=0, keepdims=True)
```

The output consists of 10 probabilities corresponding to digits:

```text
0 1 2 3 4 5 6 7 8 9
```

The class with the highest probability becomes the prediction.

### 4. Forward Propagation

```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)

    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2
```

### 5. One-Hot Encoding

Labels are converted into one-hot encoded vectors:

```python
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1

    return one_hot_Y.T
```

For example:

```text
Digit 3

[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

### 6. Backpropagation

Gradients are calculated manually using backpropagation:

```python
dZ2 = A2 - one_hot_Y

dW2 = (1 / m) * dZ2.dot(A1.T)

db2 = (1 / m) * np.sum(
    dZ2,
    axis=1,
    keepdims=True
)

dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)

dW1 = (1 / m) * dZ1.dot(X.T)

db1 = (1 / m) * np.sum(
    dZ1,
    axis=1,
    keepdims=True
)
```

### 7. Gradient Descent

The parameters are updated using:

```python
W1 = W1 - alpha * dW1
b1 = b1 - alpha * db1
W2 = W2 - alpha * dW2
b2 = b2 - alpha * db2
```

---

## Training Results

The model initially failed to learn effectively when the raw pixel values (`0–255`) were used.

After normalizing the input values to `0–1`, the model began learning effectively.

Example training progress:

```text
Iteration 130 → 85.86%
Iteration 140 → 86.16%
Iteration 150 → 86.52%
Iteration 160 → 86.85%
Iteration 170 → 87.12%
Iteration 180 → 87.35%
Iteration 190 → 87.63%
Iteration 200 → 87.81%
...
Iteration 480 → 90.42%
Iteration 490 → 90.46%
```

### Final training accuracy

**~90%+**

The model was also tested against individual handwritten digit images and successfully recognized several examples.

Example:

```text
Prediction: 3
Label:      3
```

---

## Project Structure

A possible project structure:

```text
MNIST-Digit-Recognizer/
│
├── MNIST_Digit_Recognizer.ipynb
├── README.md
├── requirements.txt
└── images/
    └── predictions/
```

If your notebook has a different filename, replace `MNIST_Digit_Recognizer.ipynb` with the actual filename.

---

## Technologies Used

- **Python**
- **NumPy**
- **Matplotlib**
- **MNIST Dataset**
- **Jupyter Notebook**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AdithyanRaji/string_neural_network.git
```

Navigate into the project:

```bash
cd string_neural_network
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the MNIST notebook and run the cells sequentially.

---

## Requirements

Create a `requirements.txt` file containing:

```text
numpy
matplotlib
jupyter
```

---

## Key Learnings

This project helped me understand neural networks beyond simply using high-level machine learning APIs.

### Concepts explored

- How images are represented as numerical matrices
- Flattening images into feature vectors
- Data normalization
- Neural network architecture
- Weight and bias initialization
- Forward propagation
- Activation functions
- ReLU
- Softmax
- One-hot encoding
- Backpropagation
- Partial derivatives and gradients
- Gradient descent
- Parameter optimization
- Model accuracy
- Training vs development data
- Individual prediction visualization

One of the most important lessons from the project was seeing how **input preprocessing directly affects model training**. Feeding the network raw pixel values resulted in the model getting stuck around random-guessing accuracy, while normalizing the pixels to `[0,1]` allowed the network to learn effectively.

---

## Limitations

This is intentionally a simple neural network designed primarily for learning.

Current architecture:

```text
784 → 10 → 10
```

Limitations include:

- Small hidden layer
- Fully connected architecture
- No convolutional layers
- No dropout
- Basic parameter initialization
- Limited optimization techniques
- Lower performance than modern CNN-based MNIST models

The purpose of the project is therefore **understanding neural-network fundamentals rather than achieving state-of-the-art performance**.

---

## Future Improvements

Possible improvements include:

- [ ] Increase hidden-layer size
- [ ] Experiment with different learning rates
- [ ] Implement better weight initialization
- [ ] Add loss tracking and visualization
- [ ] Generate a confusion matrix
- [ ] Analyze commonly misclassified digits
- [ ] Build a deeper neural network
- [ ] Implement a CNN using PyTorch
- [ ] Compare the NumPy network with a PyTorch implementation
- [ ] Build a FastAPI backend
- [ ] Create a React-based drawing interface
- [ ] Allow users to draw digits and receive predictions in real time
- [ ] Deploy the application

---

## Future Application

The eventual goal is to turn this learning project into an interactive web application:

```text
              User
                │
                ▼
       ┌─────────────────┐
       │ React Frontend  │
       │ Draw a Digit    │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ FastAPI Backend │
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │ Trained ML Model│
       └────────┬────────┘
                │
                ▼
        Prediction: 7
        Confidence: 98%
```

---

## Conclusion

This project was built to understand the **fundamentals of neural networks from the ground up**.

Rather than relying completely on a machine-learning framework, I implemented the core learning process using NumPy and gained practical experience with:

```text
Data
 ↓
Preprocessing
 ↓
Forward Propagation
 ↓
Activation Functions
 ↓
Prediction
 ↓
Backpropagation
 ↓
Gradient Descent
 ↓
Model Training
 ↓
Evaluation
```

The project provides a foundation for moving toward more advanced topics such as **deep neural networks, CNNs, PyTorch, computer vision, and model deployment**.

---

## Author

**Adithyan R.**

Computer Science & Engineering | Machine Learning | Data Science | Web Development

---

### ⭐ If you found this project useful

Feel free to explore the code, experiment with the architecture, and build upon it.
