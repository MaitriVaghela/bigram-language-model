# bigram-language-model

📚 Bigram Language Model (From Scratch)

This repository presents a from-scratch implementation of a Bigram Language Model, developed to build a deep, first-principles understanding of probabilistic language modeling and sequence generation.

⚙️ Model Formulation
The bigram assumption simplifies language modeling as:

P(x₁, x₂, ..., xₙ) = ∏ᵢ₌₁ⁿ P(xᵢ | xᵢ₋₁)

Each token depends only on the immediately preceding token.

🏗️ Implementation Details
1. Tokenization

Character-level tokenization for simplicity and interpretability

Vocabulary constructed directly from the dataset

2. Bigram Statistics

Frequency counts computed for all token pairs

Represented as a count matrix

|       | a  | b | c |
| ----- | -- | - | - |
| **a** | 10 | 5 | 2 |
| **b** | 3  | 8 | 1 |
| **c** | 0  | 2 | 6 |


Rows: previous token (xᵢ₋₁)

Columns: current token (xᵢ)

3. Probability Estimation

Counts normalized to obtain conditional probabilities:

P(xᵢ | xᵢ₋₁) = Count(xᵢ₋₁, xᵢ) / Count(xᵢ₋₁)

Example Probability Matrix:

      a       b       c
a    0.59    0.29    0.12
b    0.25    0.67    0.08
c    0.00    0.25    0.75
4. Sampling / Generation

Text is generated autoregressively: the next token is sampled from the learned bigram distribution.

Flow Diagram:

Raw Text → Tokenization → Count Matrix → Probability Matrix → Sampling → Generated Text

Each arrow represents a processing step

Sampling step chooses the next token based on P(xᵢ | xᵢ₋₁)

📉 Limitations

Markov assumption (order = 1) restricts context

No semantic understanding beyond adjacent tokens

Struggles with long-range dependencies and structure

🔬 Future Work

This implementation can be extended in several meaningful directions:

Higher-order n-gram models

Smoothing techniques (Laplace, Kneser-Ney)

Transition to neural language models

Implementing a mini GPT-style transformer
