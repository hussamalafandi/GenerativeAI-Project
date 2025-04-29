<<<<<<< HEAD
🧠 MiniGPT: Autoregressive Language Model

🔗 Weitere Links:

https://wandb.ai/altkachenko11-hochschule-hannover/projects
https://huggingface.co/altkachenko11

📚 Projektbeschreibung

In diesem Projekt habe ich ein kleines autoregressives Decoder-Only Sprachmodell (ähnlich wie GPT) mit PyTorch gebaut und trainiert.

Ich habe zwei Aufgaben umgesetzt:

Eigenes MiniGPT Modell: selbst gebaut, trainiert und auf Hugging Face hochgeladen.

Feintuning von GPT-2: ein vortrainiertes GPT-2 Modell auf den Tiny Shakespeare Datensatz angepasst.

Das Projekt erfüllt alle Anforderungen aus dem Kursauftrag.

🚀 Modell 1: Eigenes MiniGPT

✨ Details:

Architektur: Transformer Decoder mit 2 Layern, 4 Attention-Heads

Training: Auf Tiny Shakespeare Datensatz

Tokenizer: GPT-2 Tokenizer (AutoTokenizer)

Optimierer: Adam

Loss Function: Cross Entropy Loss (mit Padding Ignorierung)

Training Loop: komplett von Hand geschrieben

Logging: via Weights & Biases (wandb)

Parameterzahl: < 1 Million (trainierbar auf CPU)

🔥 Training:

5 Epochen

Trainings- und Validierungsloss werden nach jeder Epoche geloggt

Loss sichtbar gesunken

🛠️ Wichtige Dateien:

MiniGPT-Modell: altkachenko11/my-mini-gpt

🚀 Modell 2: Feintuning GPT-2

✨ Details:

Basismodell: GPT-2 (gpt2 von Hugging Face)

Training: Auf Shakespeare-Datensätzen

Framework: Hugging Face Trainer API

Logging: ebenfalls via wandb

Tokenizer: GPT2Tokenizer

Loss: automatisch durch Trainer API

Callbacks: eigener W&B Callback für Evaluation Loss

🔥 Training:

4 Epochen

Evaluation alle 20 Schritte

Verlust sinkt während des Trainings

🛠️ Wichtige Dateien:

Fine-Tuned GPT-2 Modell: altkachenko11/gpt2-finetuned

🛒 Genutzte Tools:

PyTorch

Hugging Face Transformers

Weights & Biases (wandb)

Datasets: Tiny Shakespeare

🛠️ Beispiel: Textgenerierung mit MiniGPT:

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("altkachenko11/my-mini-gpt")
model = AutoModelForCausalLM.from_pretrained("altkachenko11/my-mini-gpt")

prompt = "Hallo, mein Name ist"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    top_k=50,
    temperature=0.9
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)


📦 Veröffentlichung:

Beide Modelle wurden erfolgreich auf den Hugging Face Model Hub hochgeladen.

Model Card und Tags hinzugefügt

Modelle sind öffentlich verfügbar

✅ Erfüllt:

Modell bauen

Eigenen Trainingsloop schreiben

Logging mit Weights & Biases

Modell auf Hugging Face hochladen

Pull Request im eigenen Branch erstellen

README.md schreiben

🎉 Vielen Dank fürs Anschauen!

=======
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
>>>>>>> f57bf29360e79707cf126d7601fae3c338c0c8b9
