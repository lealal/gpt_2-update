# Modern GPT-2 Small Reimplementation (116M Parameters)

This project reimplements a **GPT-2 Small–scale language model (~116M parameters)** using **modern transformer design practices**, inspired by the book **_LLMs From Scratch_ by Sebastian Raschka**.

The goal of this project is not to compete with state-of-the-art models, but to:

- Rebuild a GPT-2–scale model from scratch
- Integrate **modern architectural improvements**
- Train on a **larger open dataset**
- Run the full training pipeline on **consumer hardware**

This repository demonstrates the **complete lifecycle of training a language model**:

1. Dataset preparation  
2. Tokenization and efficient storage  
3. Model architecture implementation  
4. Training loop and learning rate scheduling  
5. Evaluation and text generation

---

# Project Overview

The final model:

| Property | Value |
|--------|------|
| Parameters | ~116M |
| Context Length | 512 |
| Layers | 10 |
| Attention Heads | 8 |
| KV Groups | 4 |
| Embedding Dimension | 720 |
| Hidden Dimension | 1280 |
| Training Tokens | ~500M |
| Dataset | RedPajama |
| Hardware | Apple M3 Pro (MPS) |

---

# Key Architectural Changes vs GPT-2

While the model targets **GPT-2 Small scale**, several **modern transformer improvements** were introduced.

## 1. RoPE instead of positional embeddings

GPT-2 uses learned positional embeddings.  
This implementation replaces them with **Rotary Positional Embeddings (RoPE)**.

Benefits:

- Better extrapolation to longer contexts
- Parameter-free positional encoding
- Used in modern architectures such as **LLaMA**

Reference:  
RoFormer – Enhanced Transformer with Rotary Position Embedding

---

## 2. Grouped Query Attention (GQA)

Instead of standard **Multi-Head Attention**, this model uses **Grouped Query Attention**.

Advantages:

- Reduced memory usage
- Faster inference
- Used in modern LLMs such as **LLaMA-2 and LLaMA-3**

---

## 3. RMSNorm instead of LayerNorm

The model replaces LayerNorm with **RMSNorm**, which:

- Is computationally cheaper
- Improves numerical stability
- Is widely used in modern LLM architectures

---

## 4. QK Normalization

Queries and keys are normalized before computing attention scores.  
This improves **training stability** and helps prevent extremely large attention logits.

---

## 5. Modern MLP Block (SwiGLU-style)

The feedforward block uses:

- **SiLU activation**
- **Gated linear projection**

This structure is similar to the **SwiGLU feedforward networks** used in many modern LLMs.

---

## 6. No Dropout

Dropout layers were removed entirely.

Modern LLMs often rely on:

- large datasets
- normalization layers
- optimizer regularization

instead of dropout for regularization.

---

# Dataset

The model is trained using **RedPajama-Data-V2**.

Dataset source:  
https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2

### Filtering

To improve dataset quality, only documents satisfying the following conditions were used:

- English language
- Partition: `head_middle`
- Minimum length: 250 characters

### Dataset Size

The project targets approximately:

| Split | Tokens |
|------|------|
| Train | ~500M |
| Validation | ~39M |
| Test | ~39M |

Documents are tokenized using the **GPT-2 tokenizer**.

Since the dataset paper reports an average of **~1500 tokens per document**, approximately **750k documents** are sampled.

---

# Efficient Token Storage

Tokenized data is stored in **binary format** using NumPy memmap files:

```
train.bin
val.bin
test.bin
```

Advantages:

- Memory efficient
- Fast loading
- Scales well for large datasets

Each training sample consists of:

```
input_tokens
target_tokens
```

Where the target represents the **next token prediction task**.

---

# Model Architecture

```
Token Embedding
      │
      ▼
10 × Transformer Blocks
      │
      ├── RMSNorm
      ├── Grouped Query Attention + RoPE
      ├── Residual Connection
      ├── RMSNorm
      ├── SwiGLU-style MLP
      └── Residual Connection
      │
      ▼
Final RMSNorm
      │
      ▼
Linear Output Head
```

Total parameters:

```
~116M
```

---

# Training

Training was performed on a **consumer laptop (Apple M3 Pro)** using the **PyTorch MPS backend**.

### Training Configuration

| Parameter | Value |
|------|------|
| Epochs | 1 |
| Batch Size | 8 |
| Gradient Accumulation | 2 |
| Effective Batch Size | 16 |
| Warmup | 5% |
| Optimizer | AdamW |
| Peak Learning Rate | 3e-4 |
| Initial LR | 3e-5 |
| Minimum LR | 3e-5 |

### Learning Rate Schedule

The training loop uses:

1. **Linear warmup**
2. **Cosine decay**

This is a standard schedule used in many modern LLM training pipelines.

---

# Evaluation

Loss curves show **stable training behavior**.

Training and validation losses remain close throughout training, indicating **minimal overfitting** and good generalization.

---

# Results

Final evaluation on the test set:

| Metric | Value |
|------|------|
| Test Loss | ~4.41 |
| Perplexity | ~82 |

For a **116M parameter model trained on ~500M tokens**, this result is within the expected range.

---

# Text Generation

Example prompts used for generation:

```
Dedication will always pay
Large language models learn by
```

The model is capable of generating:

- Grammatically correct sentences
- Locally coherent short passages

However, the model occasionally produces **repetitive patterns**, which is common for smaller language models trained on limited data.

### Sampling Methods Implemented

- Greedy decoding
- Temperature sampling
- Top-k sampling

Example configuration:

```
temperature = 3.0
top_k = 20
```

These techniques improve diversity and reduce deterministic repetition.

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Prepare the Dataset

Run the dataset preparation cells to:

- download RedPajama samples
- filter documents
- tokenize text
- store tokens as binary files

Output files:

```
train.bin
val.bin
test.bin
```

---

## Train the Model

The training methods will:

- train the model
- periodically evaluate validation loss
- generate sample text
- save model checkpoints

---

## Evaluate the Model

Evaluation computes:

- test loss
- perplexity

---

# Hardware Used

Training was performed on:

```
Apple MacBook Pro
M3 Pro
PyTorch MPS backend
```

The project was intentionally designed to be **reproducible on consumer hardware** without requiring GPUs or distributed clusters.

---

# Future Improvements

Possible extensions include:

- Training on **larger token budgets (1-5B tokens)**
- Increasing context length to **1024+**
- Implementing **Flash Attention**
- Adding **weight tying**
- Using **mixed precision training**
- Scaling the architecture to **GPT-2 Medium or Large**
- Implementing **KV cache for faster inference**

---

# References

- Sebastian Raschka — *LLMs From Scratch*  
- GPT-2 Paper (Radford et al., 2019)  
- RoFormer: Rotary Position Embedding  
- RedPajama Dataset  

---

# Conclusion

This project demonstrates that a **modernized GPT-2–scale transformer can be implemented and trained from scratch on consumer hardware**.

Despite the relatively small training dataset, the model successfully learns useful linguistic patterns and generates coherent text. The experiment highlights the core components required to build and train modern language models while incorporating several architectural improvements used in current LLM systems.