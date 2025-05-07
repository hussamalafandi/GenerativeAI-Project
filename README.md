🧠 MiniGPT: Autoregressive Language Model
=======


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
=======
>>>>>>> ca60a7dc7597e825a9917490aa61fccc09869101
# 🧠 Abschlussprojekt: Entwicklung eines eigenen Sprachmodells
>>>>>>> aed5f8490a98a9cc8a011cfb482eb175b6e99455

- [Hugging Face](https://huggingface.co/altkachenko11)
- [WandB](https://wandb.ai/altkachenko11)

📚 Projektbeschreibung

In diesem Projekt habe ich ein kleines autoregressives Decoder-Only Sprachmodell mit PyTorch gebaut und trainiert.

Ich habe zwei Aufgaben umgesetzt:

1. Eigenes MiniGPT Modell:  gebaut, trainiert und auf Hugging Face hochgeladen.

2. Feintuning von GPT-2: ein vortrainiertes GPT-2 Modell auf den Tiny Shakespeare Datensatz angepasst.

Das Projekt erfüllt alle Anforderungen aus dem Kursauftrag.

🚀 Modell 1: Eigenes MiniGPT

✨ Details:

Architektur: Transformer Decoder mit 2 Layern, 4 Attention-Heads nhead=4 → 4 attention-heads

num_layers=2 → 2 Transformer.

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

🛒 Genutzte Tools
PyTorch

Hugging Face Transformers

Weights & Biases (wandb)

Datasets: Tiny Shakespeare

🛠️ Beispiel: Textgenerierung mit MiniGPT

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

📦 Veröffentlichung
Beide Modelle wurden erfolgreich auf den Hugging Face Model Hub hochgeladen

Model Card und Tags hinzugefügt

Modelle sind öffentlich verfügbar.


✅ Erfüllt:
 Modell bauen

 Eigenen Trainingsloop schreiben

 Logging mit Weights & Biases

 Modell auf Hugging Face hochladen

 Pull Request im eigenen Branch erstellen

 README.md schreiben


 🎉 Vielen Dank fürs Anschauen!


Viel Erfolg! 🚀



