# 🧠 Abschlussprojekt: Entwicklung eines eigenen Sprachmodells

Model Card.md (English version)
 
---
license: apache-2.0
language: en
tags:
  - text-generation
  - autoregressive
  - decoder-only
  - gpt-style
  - shakespeare
library_name: pytorch
---

# Shakespeare GPT-Style Transformer (Small)

## Model Description

This is a **decoder-only Transformer** (GPT-style) language model trained on Shakespearean texts using **PyTorch**.

Key characteristics:
- **Architecture**: TransformerDecoder-based, similar to GPT models.
- **Training Objective**: Autoregressive next-token prediction using **CrossEntropyLoss**.
- **Embedding Dimension**: 384
- **Number of Layers**: 8
- **Number of Attention Heads**: 8
- **Feedforward Dimension**: 2048
- **Dropout**: 0.1
- **Maximum Sequence Length**: 128 tokens
- **Training Dataset**: A curated collection of Shakespearean plays.

## Intended Use

- Text generation in Shakespearean style
- Educational purposes (learning how GPT models work)
- Experiments with small GPT-style architectures

## How to Use

```python
import torch
from model import MyTransformerModel  # replace with your model's loading function
from tokenizer import tokenizer       # replace with your tokenizer if needed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = MyTransformerModel(...)
model.load_state_dict(torch.load("path_to_model.pt"))
model.eval()
model.to(device)

# Tokenize
input_text = "I swear by the name of the king"
inputs = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# Generate (autoregressive sampling)
with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=70, temperature=1.0, top_k=60)

print(tokenizer.decode(outputs[0]))

Training Details
The model was trained for 15 epochs with progressive learning rate decay.
Checkpointing was done at every epoch to allow fine selection of the best model.

Evaluation
Final metrics (after epoch 5, selected model):

Train Loss: ~3.72

Validation Loss: ~4.38

Limitations
The model generates Shakespearean-style English, but it is not a factual model.

Small architecture, not suitable for long or very coherent generation.

No fine-tuning on modern English.

License
Apache 2.0 License.

Acknowledgments
Inspired by OpenAI's GPT architecture and Hugging Face Transformers.

Thanks for reviewing this model! 🚀
