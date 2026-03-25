# Bigram Language Model

A character-level **bigram language model** built from scratch in PyTorch. The model predicts the next character in a sequence using only the immediately preceding character — making it the simplest form of a language model.

---

## Overview

A **bigram model** works with 2 characters at a time: given the current character, it predicts the next one. This implementation takes a names dataset as input, trains a neural network to learn bigram probabilities, and generates new name-like sequences by sampling from the learned distribution.

**Pipeline:**

```
TRAINING DATA          TRAINING                  TRAINED MODEL
(list of names)  →  count bigrams →          (probability table P)
                    build N →
                    convert to P
                          ↓
                       SAMPLING
                  (generate new names using P)
                          ↓
                        OUTPUT
                  ("emma", "olivia", "aria" ...)
```

---

## How It Works

### 1. Dataset & Vocabulary

The model trains on a dataset of names (`names_makemore.txt`). Each name is wrapped with a `.` token that acts as both the start and end-of-word marker.

Characters are mapped to integer indices, with `.` assigned to index `0` and `a–z` mapped to `1–26`.

```python
sToI = {s: i+1 for i, s in enumerate(chars)}
sToI['.'] = 0   # start/end token at index 0
```

### 2. Building the Training Set

Bigram pairs `(ch1, ch2)` are extracted from every consecutive character pair in each word, including the boundary tokens.

```python
# Example: "emma" → (.e), (em), (mm), (ma), (a.)
for ch1, ch2 in zip(chs, chs[1:]):
    xs.append(sToI[ch1])
    ys.append(sToI[ch2])
```

**Total training examples:** 228,146 bigrams

### 3. Neural Network

The model is a single linear layer — a 27×27 weight matrix `W` — that maps one-hot encoded input characters to logits over the 27-character vocabulary.

```
Input character (one-hot, 27-dim)
        ↓
   Linear layer W (27×27)
        ↓
      Logits
        ↓
    exp() → Softmax
        ↓
  Probability distribution over next character
```

```python
xenc = F.one_hot(xs, num_classes=27).float()  # one-hot encoding
logits = xenc @ W                              # linear layer
counts = logits.exp()                          # exp of logits
probs = counts / counts.sum(1, keepdims=True)  # softmax → probabilities
```

### 4. Loss Function

The model is trained by minimizing the **negative log-likelihood (NLL)** of the correct next character, with **L2 regularization** on the weights to smooth the probability distribution:

```python
loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W**2).mean()
```

### 5. Training

Trained with **gradient descent** for 100 steps using a learning rate of 50:

```python
for k in range(100):
    # Forward pass
    logits = xenc @ W
    probs  = softmax(logits)
    loss   = NLL + regularization

    # Backward pass
    W.grad = None
    loss.backward()

    # Weight update
    W.data += -50 * W.grad
```

**Loss progression:** starts at ~3.77 and converges to ~2.49 after 100 steps.

### 6. Sampling

After training, new names are generated **autoregressively**:
- Start with the `.` token (index 0)
- Sample the next character from the predicted probability distribution
- Feed it back as the next input
- Stop when `.` is sampled again

**Sample generated names:**
```
cexze
momasurailezityha
konimittain
llayn
ka
```

---

## Model Details

| Component         | Details                                  |
|-------------------|------------------------------------------|
| Model type        | Single linear layer (bigram neural net)  |
| Vocabulary size   | 27 (a–z + `.` start/end token)           |
| Context window    | 1 character (bigram)                     |
| Weight matrix     | W: (27, 27)                              |
| Input encoding    | One-hot (27-dim)                         |
| Output            | Softmax probability over 27 characters   |
| Loss              | Negative log-likelihood + L2 regularization |
| Optimizer         | Gradient descent (lr = 50)               |
| Training steps    | 100                                      |
| Training examples | 228,146 bigrams                          |
| Framework         | PyTorch                                  |

---

## Project Structure

```
bigram-language-model/
│
└── bigram-language-model.ipynb   # Full implementation: data prep, training, evaluation, sampling
```

---

## Key Concepts

- **Bigram model** — predicts the next character using only the current character as context
- **One-hot encoding** — each input character is represented as a sparse 27-dimensional vector
- **Softmax** — converts raw logits into a valid probability distribution via `exp()` + normalization
- **Negative log-likelihood** — measures how well the model predicts the correct next character; lower is better
- **L2 regularization** — penalizes large weights to smooth the learned probability table and prevent overfitting
- **Autoregressive sampling** — generates sequences one character at a time by repeatedly sampling from the output distribution

---

## References

- [Andrej Karpathy — makemore series](https://github.com/karpathy/makemore)
- [Andrej Karpathy — Neural Networks: Zero to Hero (YouTube)](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

