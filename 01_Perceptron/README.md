# Module 1, Model 1: The Perceptron

## Introduction

This project is a from-scratch implementation of the **Perceptron**, one of the earliest and simplest types of artificial neural networks. Invented by Frank Rosenblatt in 1958, the Perceptron is a single-neuron model that can learn to solve binary classification problems. Its invention was a pivotal moment, challenging the purely symbolic approaches to AI at the time and laying the groundwork for the field of neural networks.

This implementation demonstrates the Perceptron's core learning algorithm on three classic datasets to showcase both its historical capabilities and its fundamental limitations.

## Core Innovation

The core innovation of the Perceptron is its simple, error-driven learning rule. The model only updates its weights when it makes a mistake. If a prediction is correct, the model remains unchanged. If it's incorrect, the weights are adjusted in the direction that would make the prediction more accurate. This elegant algorithm allows the model to progressively "learn" a decision boundary from labeled examples.

## How to Run This Code

### Prerequisites
* Python 3.8+
* Git

### Installation
1.  Clone the repository and navigate to the project directory.
2.  Create and activate a Python virtual environment.
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Training & Evaluation
The main training script can run three different experiments using a command-line argument. All outputs (logs and visualizations) are saved to the `/outputs` directory.

* **Run the AND gate experiment (Success Case):**
    ```bash
    python src/train.py --experiment and
    ```
* **Run the XOR gate experiment (Failure Case):**
    ```bash
    python src/train.py --experiment xor
    ```
* **Run the MNIST (0 vs 1) experiment (Success Case):**
    ```bash
    python src/train.py --experiment mnist
    ```

## Results and Analysis

We conducted experiments to demonstrate both the Perceptron's strengths and its critical weaknesses.

### "Success" Case 1: The AND Gate
As a baseline, the Perceptron was trained on the simple, linearly separable AND gate data. As expected, it achieved **100% accuracy**, quickly finding a decision boundary that perfectly separates the classes. This demonstrates the algorithm's guaranteed convergence on linearly separable problems.

| Learning Curve | Decision Boundary |
| :---: | :---: |
| ![AND Learning Curve](outputs/visualizations/learning_curve_and.png) | ![AND Decision Boundary](outputs/visualizations/decision_boundary_and.png) |

### "Success" Case 2: MNIST (0 vs 1)
To test the model on a real-world problem, it was trained to differentiate between handwritten '0's and '1's from the MNIST dataset. It achieved an impressive **99.94% accuracy**. This is possible because, in the high-dimensional space of pixel data, these two digits are largely linearly separable. The learned weights visualization clearly shows the model learning a "template" for the digit '1' (a central vertical line).

| Learning Curve                                                    | Learned Weights                                                      |
| :---------------------------------------------------------------- | :------------------------------------------------------------------- |
| ![MNIST Learning Curve](outputs/visualizations/learning_curve_mnist.png) | ![MNIST Learned Weights](outputs/visualizations/perceptron_weights_mnist.png) |

### The "Failure" Case: XOR
The Perceptron fails completely on the XOR problem, achieving only **50% accuracy** (equivalent to random guessing). The XOR data is not linearly separable, meaning a single straight line cannot divide the classes. The learning curve shows that the model never converges, and the decision boundary plot visually confirms the impossibility of finding a valid linear solution. This failure is the model's key limitation.

| Learning Curve                                                  | Decision Boundary                                                      |
| :-------------------------------------------------------------- | :--------------------------------------------------------------------- |
| ![XOR Learning Curve](outputs/visualizations/learning_curve_xor.png) | ![XOR Decision Boundary](outputs/visualizations/decision_boundary_xor.png) |


## Key Takeaways
* **The Perceptron is a Linear Classifier**: The model's fundamental design allows it to find a linear boundary (a line in 2D, a plane in 3D, or a hyperplane in higher dimensions) to separate two classes.
* **The Learning Rule is Error-Driven**: The Perceptron's weights are only updated when a classification error occurs. If the prediction is correct, no learning happens.
* **Success Depends on Linear Separability**: The Perceptron is guaranteed to find a solution for datasets that are linearly separable. We proved this with the AND gate and the MNIST '0' vs '1' task.
* **Failure is Guaranteed for Non-Linear Data**: The model's greatest weakness is that it fundamentally cannot solve problems that are not linearly separable, with the XOR gate being the classic example. This limitation was a major driver for future research in the field.
* **High Dimensionality Can Be Deceiving**: A problem that seems complex (like classifying images) can sometimes be linearly separable in a high-dimensional space (like 784-dimensional pixel space), making it solvable by a simple model like the Perceptron.

## Detailed Documentation

For a deeper dive into the theory and our findings, please see the detailed documents in the `/docs` directory.

* **[Theoretical Deep Dive](docs/01_deep_dive.md)**: An exploration of the model's history, architecture, and the mathematical theory behind its innovation.
* **[Empirical Analysis](docs/02_empirical_analysis.md)**: A detailed report of our training runs, performance metrics, and an analysis of the model's strengths and weaknesses.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.