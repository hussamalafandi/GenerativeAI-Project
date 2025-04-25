# 📜 Tiny Shakespeare GPT — Text Generation Project

Welcome to my **Tiny Shakespeare GPT** project!  
This project is a decoder-only language model inspired by GPT architecture. It’s trained on the Tiny Shakespeare dataset and built using **[PyTorch](https://pytorch.org/)**, **[Hugging Face Transformers](https://huggingface.co/transformers/)**, and **[Weights & Biases (wandb)](https://wandb.ai/)** for experiment tracking.

---

## 🚀 Project Goal

The goal of this project is to:

- Train a character-level language model from scratch on Shakespearean text.
- Use GPT-style architecture for text generation.
- Log and monitor training using [Weights & Biases]( https://wandb.ai/juliennemizero1-hsh/tiny-shakespeare-decoder-only/runs/sqys5sbv).
- Upload the final model to [Hugging Face](https://huggingface.co/JulienneMizero/decoder-gpt-julienne) .

---

## 📚 Dataset

I used the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) — a 1MB text file containing excerpts from Shakespeare’s works.

It’s small, perfect for training a compact model for demonstration and learning purposes.

---

## 🏗️ Model Architecture

- **Decoder-only transformer**
- Custom configuration using PyTorch
- Tokenization done using `GPT2TokenizerFast`
- Train

---

## ⚙️ Configuration

```python
block_size = 128  
batch_size = 32  
embed_dim = 128  
num_heads = 4  
num_layers = 2  
dropout = 0.1  
epochs = 6 
learning_rate = 1e-3 


