# Abschlussprojekt: Entwicklung eines eigenen Sprachmodells

Dieses Projekt implementiert ein kleines autoregressives Sprachmodell mit PyTorch, basierend auf dem Decoder-Teil des Transformer-Modells.

## 📚 Datensatz
- [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

## 🧠 Modellarchitektur
- TransformerDecoder (2 Layer, 4 Heads, d_model=64)
- Positionale Einbettungen
- Vokabulargröße basierend auf GPT2Tokenizer

## 🔧 Training
- 3 Epochen
- Verlustfunktion: CrossEntropy
- Optimierer: Adam (lr=3e-4)
- Eingabesequenzlänge: 64 Token
- Batchgröße: 32
- Validierung nach jeder Epoche

## 📈 Logging
Trainings- und Validierungsverluste wurden mit [Weights & Biases (wandb)](https://wandb.ai/) protokolliert.

## ☁️ Verfügbarkeit
Das Modell ist auf dem [Hugging Face Hub](https://huggingface.co/kullaniciadi/sprachmodell-final) verfügbar.

## ▶️ Ausführung
```bash
pip install -q transformers datasets wandb
