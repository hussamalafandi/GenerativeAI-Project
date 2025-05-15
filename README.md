# Decoder-Only Transformer Small

Ein minimalistisches autoregressives Transformer-Decoder-Modell (GPT-Stil) mit weniger als 1 M Parametern, implementiert in PyTorch und trainiert auf Tiny Shakespeare.

## Details

- **Parameter:** < 1 M  
- **Architektur:** Decoder-only (GPT-Style)  
- **Schichten:** 1  
- **Hidden Size:** 64  
- **Attention Heads:** 4  
- **Vokabular:** GPT-2 (50257 Tokens)  
- **Maximale Sequenzlänge:** 64  

## Projektstruktur

├─ .env.example # Vorlage für Environment-Variablen
├─ .gitignore # ignorierte Dateien/Ordner
├─ requirements.txt # Liste aller Abhängigkeiten
├─ train_model1.py # Trainings-Skript
└─ README.md # Projektbeschreibung

## Installation & Einrichtung

1. **Repository klonen**  
   ```bash
   git clone https://github.com/Volodymyr10105/tiny-shakespeare
   cd tiny-shakespeare

2. Abhängigkeiten installieren

pip install -r requirements.txt

3. Environment-Variablen konfigurieren

cp .env.example .env
# .env befüllen mit:
# WANDB_API_KEY=dein_wandb_key
# HUGGINGFACE_TOKEN=dein_hf_token

Training starten

python train_model1.py

Trainings- und Validierungs-Metriken werden in Weights & Biases angezeigt.

Am Ende wird das Modell automatisch in dein Hugging Face Repo hochgeladen.


Inferenz (Textgenerierung)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "invest-ua1/decoder-only-transformer-small"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(model_name)

input_text = "To be or not to be"
inputs     = tokenizer(input_text, return_tensors="pt")
outputs    = model.generate(inputs.input_ids, max_length=50)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

https://wandb.ai/invest-ua1-self/transformer_decoder_only_final/runs/sav6fkg8
wandb\run-20250514_013355-sav6fkg8\logs
https://huggingface.co/invest-ua1/decoder-only-transformer-small
Lizenz
MIT © Volodymyr# tiny-shakespeare
# tiny-shakespeare
Пояснение к коду (русский)
Загрузка переменных окружения

python
Копировать
Редактировать
from dotenv import load_dotenv
load_dotenv()
— позволяет считывать секретные ключи (W&B и Hugging Face) из файла .env.

Импорты и настройки

torch, nn, Dataset, DataLoader — базовые компоненты PyTorch.

AutoTokenizer из transformers — токенизатор GPT-2.

wandb — для логирования метрик в Weights & Biases.

Задаются гиперпараметры (размер словаря, размер модели, число голов, длины последовательностей, количество эпох и т.п.).

Инициализация W&B

python
Копировать
Редактировать
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="transformer_decoder_only_final")
— логинимся и создаём новый эксперимент.

Токенизация и загрузка данных

python
Копировать
Редактировать
url = "https://raw.githubusercontent.com/karpathy/char-rnn/.../input.txt"
text = requests.get(url).text
tokens = tokenizer.encode(text)
— скачиваем Tiny-Shakespeare, кодируем в числовую последовательность.

Класс датасета

python
Копировать
Редактировать
class TextDataset(Dataset):
    def __init__(...):
        # разбиваем токены на куски длины seq_len+1
Каждый элемент — тензор из seq_len+1 токенов: первые seq_len — вход, сдвинутые на 1 к правому краю — цель.

Модель — decoder-only Transformer

python
Копировать
Редактировать
class DecoderOnlyModel(nn.Module):
    self.embedding, self.pos_embedding
    decoder_layer = nn.TransformerDecoderLayer(...)
    self.decoder = nn.TransformerDecoder(...)
    self.output = nn.Linear(...)
— GPT-style: эмбеддинги + позиционные эмбеддинги, causal-маска, линейный выход.

Функции train/validate

В train() — прямой проход, вычисление потерь, оптимизация, логирование шаговых и эпохальных значений в W&B.

В validate() — аналогично, но без градиентов, для контроля качества на валидации.

Генерация текста

python
Копировать
Редактировать
def generate_text(...):
    # autoregressive loop с causal-маской и sampling
— на входе «шаблонная» строка, на выходе — продолжение.

Логин и загрузка на Hugging Face Hub

python
Копировать
Редактировать
from huggingface_hub import login, create_repo, upload_folder
login(token=os.getenv("HUGGINGFACE_TOKEN"))
create_repo(...)
upload_folder(...)
— сохраняем веса (pytorch_model.bin), конфиг и токенизатор, закачиваем в ваш репозиторий на hub.

Erläuterung zum Code (Deutsch)
Environment-Variablen laden

python
Копировать
Редактировать
from dotenv import load_dotenv
load_dotenv()
— lädt API-Tokens aus der Datei .env.

Imports und Hyperparameter

PyTorch: torch, nn, Dataset, DataLoader.

Hugging Face Tokenizer: AutoTokenizer.

Weights & Biases: wandb.

Definition der Modell- und Trainingsparameter (Vokabulargröße, Modell-Dimension, Anzahl Transformer-Layer, Sequenzlänge, Epochen usw.).

W&B Setup

python
Копировать
Редактировать
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="transformer_decoder_only_final")
— Authentifizierung und Initialisierung des Experiments.

Tokenisierung & Datendownload

python
Копировать
Редактировать
url = "https://raw.githubusercontent.com/karpathy/char-rnn/.../input.txt"
text = requests.get(url).text
tokens = tokenizer.encode(text)
— Tiny Shakespeare Text herunterladen und in Token-IDs umwandeln.

Dataset-Klasse

python
Копировать
Редактировать
class TextDataset(Dataset):
    # erstellt Sequenzen der Länge seq_len+1 als Input–Target-Paare
— jeder Datensatz-Eintrag besteht aus Input-Sequenz und dem nächsten Token als Ziel.

Decoder-Only Transformer Modell

python
Копировать
Редактировать
class DecoderOnlyModel(nn.Module):
    # Embeddings + Positional Embeddings
    # TransformerDecoder mit causal Mask
    # Lineares Output-Layer
— GPT-ähnliche Architektur für autoregressives Text-Generieren.

Trainings- und Validierungsfunktionen

train(): Forward-Pass, Loss-Berechnung, Backprop, Optimizer-Step, W&B-Logging.

validate(): Evaluation ohne Gradient, Berechnung des Validierungs-Loss.

Text-Generierung

python
Копировать
Редактировать
def generate_text(...):
    # autoregressives Sampling mit Temperatur
— generiert fortlaufenden Text basierend auf einem Prompt.

Upload zum Hugging Face Hub

python
Копировать
Редактировать
login(token=os.getenv("HUGGINGFACE_TOKEN"))
create_repo(repo_id, private=False)
upload_folder(folder_path=save_dir, repo_id=repo_id, commit_message=...)
— speichert Modell-Gewichte, Konfig und Tokenizer und lädt sie in Ihr HF-Repository.


