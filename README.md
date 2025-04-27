# 🧠 Abschlussprojekt: Entwicklung eines eigenen Sprachmodells

In diesem Projekt setzt du dein Wissen über Sprachmodelle in die Praxis um und entwickelst dein eigenes autoregressives Modell auf Basis von PyTorch. Zusätzlich lernst du Tools wie Weights & Biases (wandb) und den Hugging Face Model Hub kennen – genau wie im echten ML-Workflow.

---

## ✅ Projektanforderungen

### 1. Modell
- Erstelle ein **Decoder-only Sprachmodell** mit Modulen aus `torch.nn`.
- z. B. `nn.TransformerDecoder`, `nn.TransformerDecoderLayer` usw. verwenden.
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



## 🌟


- Lade ein GPT-2-Modell und den passenden Tokenizer (`GPT2Tokenizer`) mit `from_pretrained`.
- Trainiere es auf deinem Datensatz mit der `Trainer` API.
- Logge mit wandb und lade auch dieses Modell auf Hugging Face hoch.


