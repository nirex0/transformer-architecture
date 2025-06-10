# "Attention Is All You Need": A PyTorch Implementation

[](https://www.google.com/search?q=%5Bhttps://opensource.org/licenses/MIT%5D\(https://opensource.org/licenses/MIT\))

This repository contains a from-scratch implementation of the original Transformer architecture, as introduced in the seminal paper **["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)** by Vaswani et al.

The goal of this project is to provide a clear, modular, and heavily commented codebase that demystifies the components of the Transformer. It is designed specifically for educational purposes, allowing students, researchers, and developers to see how each part of the architecture is built and how they fit together.

## The Transformer Architecture

This implementation faithfully follows the original architectural diagram presented in the paper. The model is composed of a stacked Encoder and Decoder, each containing multiple identical blocks.

![The Transformer Model Architecture](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

-----

## Key Features 📜

  * **From-Scratch & Modular:** Every component—from `SelfAttention` to the `EncoderBlock`—is built from basic PyTorch modules for maximum clarity.
  * **Faithful to the Paper:** The architecture adheres to the original design, including multi-head attention mechanisms and the encoder-decoder stack.
  * **Pre-Layer Normalization:** This implementation uses the Pre-Norm formulation (`LayerNorm(x)` then `Sublayer(x)`), which has been found to provide more stable training than the original Post-Norm version.
  * **Heavily Commented:** The code is documented to explain the *why* behind each step and the flow of tensors.
  * **Self-Contained:** A single Python file contains all the necessary building blocks and a runnable example to demonstrate instantiation and correct tensor shapes.

## How to Run

This is not a package. To run the example, save the code as a Python file (e.g., `transformer.py`) and execute it from your terminal:

```bash
python transformer.py
```

This will instantiate the `Transformer` model with the specified hyperparameters and print the shape of the output tensor, confirming the model runs correctly.

-----

## Core Concepts Explained 🧠

#### Self-Attention (Query, Key, Value)

At its core, self-attention allows the model to weigh the importance of different words in a sequence when processing a specific word. It works like a database retrieval system:

  * **Query (Q)**: The current word asking for information.
  * **Key (K)**: All other words in the sequence, acting as labels for their information.
  * **Value (V)**: The actual information content of each word.

The model learns to match the `Query` of the current word with the `Keys` of all other words to produce attention scores. These scores are then used to create a weighted sum of the `Values`, resulting in an output embedding that is rich in contextual information.

#### Multi-Head Attention

Instead of performing attention once, Multi-Head Attention runs the process multiple times in parallel (once for each "head"). Each head learns to focus on a different type of relationship or aspect of the sequence. The results are then concatenated and projected back to the original dimension, allowing the model to capture a richer set of contextual dependencies.

#### Positional Encoding (A Critical Note)

Transformers contain no recurrence or convolution, so they have no inherent sense of word order. To fix this, we must inject information about the position of each token in the sequence.

#### Masking

In the decoder, we use "masked" self-attention. During training, the decoder should only be able to see past and present tokens when predicting the next word. The look-ahead mask is a triangular matrix that sets the attention scores for all future tokens to negative infinity, effectively zeroing them out after the softmax operation. This prevents the model from cheating.
