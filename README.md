# MiniGPT — Transformer Language Model from Scratch with JAX

> A GPT-style autoregressive language model built entirely from first principles using JAX + Flax NNX. Trained on children's stories. Deployed as a live Gradio web app.

**Sample output** — prompt: *"Once upon a time a little girl"*
> *"Once upon a time a little girl named Tim who loved to play with her friends. One day, they went to the park, she saw a big house with her mom. She wanted..."*

---

## What this is

This project implements a mini GPT (Generative Pre-trained Transformer) from scratch — no HuggingFace, no pre-trained weights. Every component is built and trained manually:

- Custom Transformer architecture with Multi-Head Attention
- Token + Positional Embeddings
- Causal (masked) self-attention for autoregressive generation
- Full training loop with AdamW + cosine LR warmup
- Live story generation via a Gradio web UI



---

## Architecture

```
Input tokens (seq_len=128)
        │
        ▼
TokenAndPositionEmbedding
  ├── Token Embedding    [vocab_size=50257, embed_dim=192]
  └── Position Embedding [maxlen=128,      embed_dim=192]
        │
        ▼
TransformerBlock × 6
  ├── MultiHeadAttention (num_heads=6, causal mask)
  ├── LayerNorm + Residual
  ├── FeedForward: Linear(192→512) → GELU → Linear(512→192)
  └── LayerNorm + Residual
        │
        ▼
Linear output layer [embed_dim=192 → vocab_size=50257]
        │
        ▼
Next-token logits → temperature sampling → generated token
```

**Model config:**

| Parameter | Value |
|---|---|
| Transformer blocks | 6 |
| Embedding dimension | 192 |
| Attention heads | 6 |
| Feed-forward dim | 512 |
| Max sequence length | 128 |
| Vocabulary size | 50,257 (GPT-2) |
| Tokenizer | tiktoken BPE |

---

## Tech Stack

| Library | Role |
|---|---|
| **JAX 0.6.2** | Functional ML framework — `grad`, `jit`, `vmap` |
| **Flax NNX 0.10.7** | Neural network layers and module system |
| **Optax** | AdamW optimizer + cosine LR warmup schedule |
| **Google Grain 0.2.13** | Efficient data loading with IndexSampler |
| **tiktoken 0.7.0** | GPT-2 BPE tokenizer (vocab size 50,257) |
| **Orbax** | Model checkpointing and weight restoration |
| **Gradio 6.5.1** | Live web UI with public share link |
| **Google Colab T4** | GPU training environment |

---

## Training Results

| | Value |
|---|---|
| Dataset | TinyStories (1,000 stories) |
| Epochs | 15 |
| Batch size | 32 |
| Sequence length | 128 tokens |
| Optimizer | AdamW (peak lr=5e-4, cosine decay) |
| Hardware | Google Colab T4 GPU |
| **Initial loss** | **8.90** |
| **Final loss** | **2.00** |

Loss curve: `8.90 → 6.78 → 5.64 → 5.10 → 4.77 → 4.56 → 4.43 → 4.35 → 4.31 → 4.30 → ... → 2.00`

---

## Project Structure

```
MiniGPT-JAX/
│
├── sample.py              # JAX fundamentals: grad, jit, vmap demo
│
├── L3/
│   ├── L3.ipynb           # Data pipeline: load, tokenize, batch
│   ├── helper.py          # StoryDataset, DataLoader (Grain)
│   └── TinyStories-1000.txt
│
├── L4/
│   ├── L4.ipynb           # Model architecture + training loop
│   ├── helper.py          # MiniGPT, TransformerBlock, training utils
│   └── TinyStories-1000.txt
│
└── L5/
    ├── L5.ipynb           # Inference + Gradio UI
    └── helper.py          # generate_text(), generate_story(), Gradio app
```

**How the files connect:**

```
sample.py         → JAX warmup (standalone, not part of model)
     ↓
L3/L3.ipynb       → loads TinyStories → tokenizes → Grain DataLoader → batches (32, 128)
     ↓
L4/L4.ipynb       → builds MiniGPT → trains → saves model_checkpoint.orbax
     ↓
L5/L5.ipynb       → loads checkpoint → generate_story() → Gradio web app
```

---

## How to Run

### On Google Colab (recommended)

**1. Mount Drive and install dependencies**
```python
from google.colab import drive
drive.mount('/content/drive')

!pip install "jax[cuda12]==0.6.2" flax==0.10.7 grain==0.2.13 \
             tiktoken==0.7.0 gradio==6.5.1 orbax-checkpoint matplotlib -q
```

**2. Navigate to project**
```python
%cd /content/drive/MyDrive/LLM-jax
!ls  # should show: L2  L3  L4  L5  sample.py
```

**3. Run L3 — verify data pipeline**
```python
%cd /content/drive/MyDrive/LLM-jax/L3
# Open and run L3.ipynb
# Expected output: batches of shape (32, 128)
```

**4. Run L4 — train the model**
```python
%cd /content/drive/MyDrive/LLM-jax/L4
# Open and run L4.ipynb
# Enable GPU: Runtime → Change runtime type → T4 GPU
# Expected: loss drops from ~8.9 to ~2.0 over 15 epochs
# Saves: model_checkpoint.orbax
```

**5. Run L5 — generate stories + launch Gradio**
```python
%cd /content/drive/MyDrive/LLM-jax/L5
# Open and run L5.ipynb
# Expected: Gradio public URL printed, story generation working
```

> **Important:** Enable T4 GPU before running L4 — Runtime → Change runtime type → T4 GPU

---

## Key Concepts Implemented

**Next-token prediction (autoregressive training)**
The model receives tokens `1→127` as input and predicts tokens `2→128` as targets — the sequence shifted by 1. This single objective teaches the model to predict the future from the past, which is the entire training mechanism of GPT.

**Causal attention mask**
A lower-triangular mask prevents each token from attending to future positions. This ensures the model generates text left-to-right and can't "cheat" during training by looking ahead.

**LayerNorm + Feed Forward (the critical fix)**
The original course code had a `TransformerBlock` with only attention and a residual connection — missing both `LayerNorm` and the `FeedForward` network. Without these, loss stalled at 5.4 and the model output only repeated punctuation. Adding both components is what dropped loss to 2.0 and produced coherent sentences.

**Temperature sampling**
At inference, dividing logits by temperature before softmax controls output diversity. `temperature=0.2` → safe, deterministic text. `temperature=0.8` → creative, more varied output. One scalar controls the entire generation personality.

**JAX functional paradigm**
Unlike PyTorch, JAX has no mutable state. Transforms `grad`, `jit`, and `vmap` compose cleanly. Random keys are explicit and must be passed manually. `@nnx.jit` compiles the training step to XLA for GPU acceleration.

---

## What's Next

- [ ] Train on full TinyStories dataset (~2M stories) for significantly better output
- [ ] Add top-k and nucleus (top-p) sampling for better generation quality
- [ ] Implement gradient clipping to stabilize training
- [ ] Add a loss curve visualization to the Gradio UI
- [ ] Deploy permanently to Hugging Face Spaces (`gradio deploy`)

---

## About

Built by **Sushma V** — final year B.Tech Computer Science student at PES University, Bengaluru.



---

