
# 📎 Links

- 🤗 Hugging Face : [NataliiaM15 on Hugging Face](https://huggingface.co/NataliiaM15/decoder-shakespeare-gpt)
- 📊 W&B Project Dashboard: [Shakespeare Transformer on Weights & Biases](https://wandb.ai/basan-1994-15-hochschule-hannover/shakespeare-transformer?nw=nwuserbasan199415)


# Decoder-Only Shakespeare GPT

This is a lightweight GPT-style decoder-only transformer model trained on the Tiny Shakespeare dataset (`karpathy/tiny_shakespeare`). It uses a custom implementation in PyTorch and supports character-level text generation.

## Model Details

- **Architecture**: Decoder-only Transformer
- **Layers**: 2
- **Embedding Size**: 128
- **Heads**: 4
- **Sequence Length**: 64
- **Training Epochs**: 4
- **Tokenizer**: GPT-2 tokenizer (character-level)

## Training

Trained on the full Tiny Shakespeare dataset for 4 epochs using Adam optimizer and cross-entropy loss. Validation loss is tracked and logged using Weights & Biases (wandb).

## Usage from Hugging Face

```python
from transformers import AutoTokenizer
import torch
from model import DecoderOnlyTransformer  # custom model class

tokenizer = AutoTokenizer.from_pretrained("NataliiaM15/decoder-shakespeare-gpt")
model = DecoderOnlyTransformer(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    seq_len=64
)
model.load_state_dict(torch.load("pytorch_model.bin"))
model.eval()

# Generate text
prompt = "ROMEO:"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
# generation loop
output = input_ids
model.eval()
for _ in range(50):
    input_trim = output[:, -64:]
    with torch.no_grad():
        logits = model(input_trim)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    output = torch.cat([output, next_token], dim=1)

generated_text = tokenizer.decode(output[0])
print(generated_text)

