# NanoTransformer: Kleines Transformer Sprachmodell

Dieses Projekt umfasst das Training eines kleinen Transformer-Decoder-Modells auf dem WikiText-2-Datensatz und das Hochladen auf den Hugging Face Hub.

---

✨ Features

- Leichtgewichtige Transformer-Decoder-Architektur
- Verwendung eines Tokenizers von Hugging Face (GPT-2 Tokenizer)
- Manuelles Training mit PyTorch
- Nachverfolgung von Trainings- und Validierungsverlust
- Logging und Monitoring mit Weights & Biases (wandb)
- Unterstützung für Top-k Sampling bei der Textgenerierung
- Modell-Upload auf Hugging Face Hub

---

🛠 Verwendete Technologien

- Python
- PyTorch
- Hugging Face Datasets & Transformers
- Weights & Biases (wandb)

---

📄 Trainingsübersicht

- Trainingsdatensatz: WikiText-2 (Raw Version)
- Maximale Sequenzlänge: 64 Tokens
- Embedding-Größe (`d_model`): 128
- Anzahl der Attention-Heads (`n_heads`): 4
- Anzahl der Decoder-Schichten (`num_layers`): 2
- Optimierer: Adam
- Lernrate: 1e-4
- Batch-Größe: 32
- Trainingsdauer: 10 Epochen

---

📤 Hugging Face Hub

Das trainierte Modell ist auf dem Hugging Face Hub verfügbar:  
➡️ [NanoTransformer auf Hugging Face](https://huggingface.co/onurozdemir/nano-transformer-onur)

---

## 🚀 Beispiel: Textgenerierung

```python
def generate(model, start_token, max_len=50, temperature=0.7, top_k=50, device="cpu"):
    model.eval()
    input_ids = start_token.to(device)

    for _ in range(max_len):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature

        if top_k is not None:
            values, indices = torch.topk(next_token_logits, top_k)
            probs = torch.softmax(values, dim=-1)
            next_token = indices.gather(-1, torch.multinomial(probs, num_samples=1))
        else:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids.squeeze().tolist()


start_text = "The meaning of life is"
start_token = tokenizer.encode(start_text, return_tensors="pt").to(device)

generated_tokens = generate(model, start_token, max_len=50, temperature=0.7, top_k=30, device=device)

generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)
```
