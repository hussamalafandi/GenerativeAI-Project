## 📘 My Transformer Language Model

This is a small Transformer-based language model trained from scratch on the Wikitext-2 dataset. It was developed as part of an educational project to understand transformer architectures and language modeling.
The model learns to predict the next token in a sequence using an autoregressive approach.

## 📦 Model Details
- **Architecture**: Decoder-only Transformer (GPT-like)
- **Layers**: 6
- **Attention Heads**: 8
- **Hidden Size (Embedding Dimension)**: 512
- **Block Size (Context Window)**: 128 tokens
- **Dropout**: 0.1
- **Vocabulary Size**: 50257 (from GPT-2 tokenizer)
- **Total Parameters**: ~32M
 Estimated for 6 layers of self-attention and feedforward networks using 512-dimensional embeddings and 8 heads.

## 🔤 Tokenizer

- **Type**: `AutoTokenizer` from Transformers
- **Pretrained Model**: `gpt2`
- **Padding Token**: set to `eos_token`
- **Tokenization Method**: Byte-level BPE (as used in GPT-2)

## 📊 Training Configuration

- **Dataset**: [WikiText-2-raw-v1](https://huggingface.co/datasets/wikitext)
- **Epochs**: 5
- **Batch Size**: 32
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4
- **Scheduler**: Cosine Annealing
- **Loss Function**: CrossEntropyLoss (ignoring padding token)

Trained using Google Colab with GPU acceleration.

🧪 Evaluation

The model was evaluated on the validation split of Wikitext-2. Validation loss and training loss are tracked via [Weights & Biases](https://wandb.ai/).
Tracking run with wandb version 0.19.9
Run data is saved locally in /content/wandb/run-20250423_174822-b23nfssf
View project at (https://wandb.ai/tet-sydorenko-private_account/my-transformer-lm)
View run at (https://wandb.ai/tet-sydorenko-private_account/my-transformer-lm/runs/b23nfssf)

📎 Files Included

- `pytorch_model.bin`: Trained model weights.
- `model.py`: Model architecture
- `config.json`: Training configuration.
- `README.md`: This model card.

## 💡 How to Use

Here’s an example of how to load and use the model for generation:

```python
import torch
from model import TransformerLM  # model class
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load model (adjust path if using from HF Hub or locally)
model = TransformerLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=512,
    n_heads=8,
    n_layers=6,
    block_size=128,
    dropout=0.1
)
model.load_state_dict(torch.load("path_to_model.pth"))
model.eval()

# Generate text
prompt = "our future"
output = generate(model, tokenizer, prompt)
print(output)
```

Note: The generate function is custom, not from Transformers API.

🔒 Limitations
- Not suitable for production or safety-critical applications.
- Trained on a small subset of text data.
- Not aligned or filtered for offensive content.

🧠 Intended Use
This model is for educational and research purposes. It is not fine-tuned for production tasks such as conversation, summarization, or question answering.

✍️ Author
Developed by: [Tetiana Sydorenko]

Project HuggingFace: (https://huggingface.co/tetsydorenko/my-transformer-lm)


# 🧠 Abschlussprojekt: Entwicklung eines eigenen Sprachmodells

Willkommen zum Abschlussprojekt dieses Kurses! In diesem Projekt setzt du dein Wissen über Sprachmodelle in die Praxis um und entwickelst dein eigenes autoregressives Modell auf Basis von PyTorch. Zusätzlich lernst du Tools wie Weights & Biases (wandb) und den Hugging Face Model Hub kennen – genau wie im echten ML-Workflow.

---

## ✅ Projektanforderungen

### 1. Modell
- Erstelle ein **Decoder-only Sprachmodell** mit Modulen aus `torch.nn`.
- Du darfst z. B. `nn.TransformerDecoder`, `nn.TransformerDecoderLayer` usw. verwenden.
- Das Modell soll autoregressiv funktionieren (wie GPT).

### 2. Tokenizer
- Verwende einen Tokenizer aus der Hugging Face `transformers`-Bibliothek.
- Beispiel: `AutoTokenizer` oder `GPT2Tokenizer`.

### 3. Training
- Trainiere dein Modell für mindestens **3 Epochen** (5 empfohlen).
- Nutze einen kleinen Datensatz wie **Tiny Shakespeare**, **WikiText-2** oder einen eigenen.
- Dein Modell sollte auch auf einer CPU trainierbar sein (< 1 Mio Parameter).
- Schreibe den Trainingsloop komplett selbst in PyTorch (kein `Trainer` verwenden).

### 4. Evaluation
- Berechne nach jeder Epoche den Loss auf einem Validierungsdatensatz.
- Der Loss muss während des Trainings **sichtbar sinken**.

### 5. Logging
- Verwende [wandb](https://wandb.ai), um Trainings- und Eval-Loss zu loggen.

### 6. Veröffentlichung
- Lade dein Modell am Ende auf den [Hugging Face Model Hub](https://huggingface.co/).
- Füge eine kurze Model Card mit Beschreibung und Tags hinzu.

### 7. Abgabe
- Forke dieses Repository.
- Erstelle einen Branch mit deinem Namen, z. B. `max-mustermann-final`.
- Füge deine `.py`-Datei oder dein Jupyter-Notebook sowie eine `README.md` hinzu.
- Erstelle einen Pull Request **bis spätestens 23:59 Uhr am 25.04.2025**.

---

## 🌟 Bonus (optional)

Wenn du möchtest, kannst du zusätzlich ein vortrainiertes Modell wie GPT-2 mithilfe der Hugging Face `transformers`-Bibliothek finetunen:

- Lade ein GPT-2-Modell und den passenden Tokenizer (`GPT2Tokenizer`) mit `from_pretrained`.
- Trainiere es auf deinem Datensatz mit der `Trainer` API.
- Logge mit wandb und lade auch dieses Modell auf Hugging Face hoch.

---

## 📝 Wichtige Hinweise

- Logging mit wandb, das Hochladen auf den Hugging Face Hub und der Pull Request auf GitHub sind **Pflicht**.
- Die Modellqualität ist nicht entscheidend, aber **der Loss muss sinken**.
- Du wirst am **Montag, den 28.04.2025** dein Projekt präsentieren und deinen Code erklären.

---

Viel Erfolg! 🚀
