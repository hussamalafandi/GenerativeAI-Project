from huggingface_hub import HfApi, create_repo, upload_folder
import torch
from dotenv import load_dotenv
import os
from transformers import GPT2TokenizerFast
from model import DecoderLanguageModel  # ← passe das ggf. an deinen Pfad an

# .env laden
load_dotenv()
token = os.getenv("HF_TOKEN")
repo_name = "ahmadisakina/decoder-language-model"

def upload_to_hf(model, tokenizer, repo_name, token):
    api = HfApi()
    create_repo(repo_id=repo_name, token=token, exist_ok=True)

    model_dir = "hf_model"
    os.makedirs(model_dir, exist_ok=True)

    # Modell speichern
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))

    # Tokenizer speichern
    tokenizer.save_pretrained(model_dir)

    # Modellkarte (README.md)
    with open(os.path.join(model_dir, "README.md"), "w") as f:
        f.write(f"""# Decoder Language Model
Ein kleiner autoregressiver Decoder-only Transformer, trainiert auf Tiny Shakespeare.

## Architektur
- d_model=128, num_layers=2, nhead=4
- ~500k Parameter

## Metriken
- Loss (Train): 0.6342
- Perplexity (Train): 1.8854

## Laden
```python
from transformers import GPT2Tokenizer
import torch
from model import DecoderLanguageModel

tokenizer = GPT2Tokenizer.from_pretrained("{repo_name}")
model = DecoderLanguageModel(vocab_size=tokenizer.vocab_size, d_model=128, nhead=4, num_layers=2)
model.load_state_dict(torch.load("pytorch_model.bin"))
model.eval()
```""")

    # Hochladen
    upload_folder(
        repo_id=repo_name,
        folder_path=model_dir,
        token=token
    )

    print(f"✅ Erfolgreich hochgeladen: https://huggingface.co/{repo_name}")

# ========== Hauptteil ==========

if __name__ == "__main__":
    # Beispielmodell und Tokenizer initialisieren
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = DecoderLanguageModel(vocab_size=tokenizer.vocab_size, d_model=128, nhead=4, num_layers=2)

    # Modell hochladen
    upload_to_hf(model, tokenizer, repo_name, token)
