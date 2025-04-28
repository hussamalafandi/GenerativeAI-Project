---
license: apache-2.0
tags:
- text-generation
- transformer-decoder
- autoregressive-model
- wikitext-2
- pytorch
language:
- en
library_name: pytorch
---

# Simple Transformer Decoder Language Model

This is a simple Transformer Decoder-based **autoregressive language model** developed as part of a deep learning project.  
It was trained on the [WikiText-2](https://paperswithcode.com/dataset/wikitext-2) dataset using PyTorch, with a focus on **learning to generate English text** in an autoregressive manner (predicting the next token given previous tokens).

The model follows a **decoder-only architecture** similar to models like **GPT**, using **causal masking** to prevent attention to future tokens.

---

## ✨ Model Architecture

- **Model type**: Transformer Decoder (only decoder layers)
- **Embedding size**: 128
- **Number of attention heads**: 4
- **Number of decoder layers**: 2
- **Feed-forward hidden dimension**: 512
- **Positional Encoding**: Sinusoidal (fixed, not learned)
- **Vocabulary size**: Based on GPT-2 tokenizer (approx. 50K tokens)
- **Max sequence length**: 256 tokens
- **Dropout**: 0.1 (inside transformer layers)

---

## 📚 Dataset

- **Name**: WikiText-2
- **Size**: ~2 million tokens
- **Language**: English
- **Task**: Next-token prediction (causal language modeling)

**Dataset Link**: [WikiText-2 on Hugging Face Datasets](https://huggingface.co/datasets/wikitext)

---

## 🏋️‍♂️ Training Details

- **Optimizer**: Adam
- **Learning rate**: 5e-4
- **Batch size**: 4
- **Training epochs**: 5
- **Loss function**: CrossEntropyLoss
- **Logging**: Weights & Biases (wandb)

✅ Loss decreased successfully during training, indicating the model learned the structure of English text.

---

## 🚀 How to Use

> Note: Since this is a custom PyTorch model (not a Hugging Face PreTrainedModel), you must manually define and load it.

```python
import torch
from transformers import AutoTokenizer
from your_custom_model_code import SimpleTransformerDecoderModel  # import your model class

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("mreeza/simple-transformer-model")

# Initialize the model
model = SimpleTransformerDecoderModel(
    vocab_size=len(tokenizer),
    d_model=128,
    nhead=4,
    num_layers=2,
    max_seq_len=256
)

# Load trained weights
model.load_state_dict(torch.load("pytorch_model.bin", map_location="cpu"))
model.eval()

# Generate text
def generate_text(model, tokenizer, prompt="Once upon a time", max_length=50):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

prompt = "Once upon a time"
generated = generate_text(model, tokenizer, prompt)
print(generated)
