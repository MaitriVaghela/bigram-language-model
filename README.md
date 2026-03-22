# bigram-language-model

📚 Bigram Language Model (From Scratch)

This repository presents a from-scratch implementation of a Bigram Language Model, developed to build a deep, first-principles understanding of probabilistic language modeling and sequence generation.

⚙️ Model Formulation
The bigram assumption simplifies language modeling as:

P(x1, x2, ..., xn) = Π_{i=1}^{n} P(xi | xi-1)

Each token depends only on the immediately preceding token.
