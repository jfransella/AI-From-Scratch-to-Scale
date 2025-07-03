# Module 1, Model 2: ADALINE (Conceptual Study)

## Introduction

This document provides a conceptual analysis of ADALINE (Adaptive Linear Neuron), the second model in our historical journey. Developed by Bernard Widrow and Tedd Hoff at Stanford around 1960, ADALINE was a critical evolution of the Perceptron. While architecturally similar, it introduced a profoundly different and more powerful learning mechanism known as the **Delta Rule**.

As a **Conceptual** model in this project, our focus is on understanding its theoretical innovation rather than a full implementation.

## Core Innovation: The Delta Rule

The single most important innovation of ADALINE is its learning rule. Unlike the Perceptron, which calculates its error based on the final binary output (0 or 1), ADALINE uses the **continuous linear output** (the weighted sum of the inputs) to calculate its error term.

| Model | Error Calculation Basis | Nature of Error |
| :--- | :--- | :--- |
| **Perceptron** | `predicted_binary_output` (from step function) | Binary (Correct/Incorrect) |
| **ADALINE** | `linear_output` (from summation) | Continuous (Magnitude of Error) |

This simple change has a massive impact. By using a continuous error value, the weight updates become proportional to how "wrong" the model is. A large error results in a large weight update, and a small error results in a small one. This process is effectively an early form of **gradient descent**, as it attempts to minimize the sum-of-squared errors of the model's linear output. The result is a more stable and reliable learning process.

## The Transition Narrative

ADALINE represents a significant step forward in *how* a neuron can learn. The Delta Rule is a more powerful and mathematically robust learning mechanism than the Perceptron's.

However, ADALINE is still just a single neuron. Like the Perceptron, it can only solve problems that are **linearly separable**. It would still fail on the XOR problem for the exact same geometric reasons.

The shared weakness of these single-neuron models is what motivates the next major architectural leap in our journey: stacking neurons into layers to create the **Multi-Layer Perceptron (MLP)**. The MLP combines the power of the Delta Rule (in a generalized form called backpropagation) with a non-linear architecture, finally unlocking the ability to solve complex problems.

## Detailed Documentation

For a deeper dive into this model, see our [Theoretical Deep Dive](docs/01_deep_dive.md), and our [Empirical Analysis](docs/02_empirical_analysis.md).

## License
This project is licensed under the MIT License.