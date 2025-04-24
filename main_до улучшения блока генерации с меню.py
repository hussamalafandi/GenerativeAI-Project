# 📦 Импорты 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import requests
import os
import torch.nn.functional as F
import random

# ⚙️ Конфигурация
config = {
    "model_name": "MyTinyDecoder",
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "block_size": 96,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Using device:", config["device"])
print("CUDA available:", torch.cuda.is_available())

torch.cuda.empty_cache()

# 📥 Шаг 1: Загрузка Tiny Shakespeare
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

# ✂️ Разделение на train / val
train_text, val_text = train_test_split(text.split("\n"), test_size=0.1, random_state=42)
train_text, val_text = "\n".join(train_text), "\n".join(val_text)

# 🔠 Шаг 2: Токенизация
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

train_ids = tokenizer(train_text, return_tensors="pt")["input_ids"].squeeze()[:5000]
val_ids = tokenizer(val_text, return_tensors="pt")["input_ids"].squeeze()[:5000]

# 📚 Dataset класс
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

train_dataset = TextDataset(train_ids, config["block_size"])
val_dataset = TextDataset(val_ids, config["block_size"])
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

# 🧠 Шаг 3: Модель — простой decoder
class TinyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=48, n_heads=2, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(0.3)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0)).to(x.device)
        out = self.decoder(x, x, tgt_mask=tgt_mask)
        out = out.permute(1, 0, 2)
        return self.linear(out)

# 🧠 Генерация текста
def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.5, top_k=60):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(config["device"])

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature  # 🔥 temp scaling

            # 🔽 top_k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token = top_k_indices[0, torch.multinomial(probs, num_samples=1).item()]

        next_token = next_token.unsqueeze(0).unsqueeze(0)  # → [1, 1]
        input_ids = torch.cat([input_ids, next_token], dim=1)

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


# 🚦 Основной запуск
if __name__ == "__main__":
    mode = input("👉 Выберите режим: [train] для обучения / [gen] для генерации: ").strip().lower()

    model = TinyDecoder(vocab_size=len(tokenizer)).to(config["device"])
    model_path = "tiny_decoder.pt"

    if mode == "train":
        wandb.init(project="tiny-language-model", config=config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1)

        # 🏋️ Тренировка
        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x, y = x.to(config["device"]), y.to(config["device"])
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # 🔍 Валидация
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(config["device"]), y.to(config["device"])
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    val_loss += loss.item()
            val_loss /= len(val_loader)

            # 💡 Шаг SCHEDULER-а
            scheduler.step(val_loss)

            # 📊 wandb лог
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"]
            })

            print(f"[{epoch+1}/{config['epochs']}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        torch.save(model.state_dict(), model_path)
        print("✅ Модель сохранена.")
        wandb.finish()

    elif mode == "gen":
        if not os.path.exists(model_path):
            print("❌ Модель не найдена. Сначала обучите её (режим train).")
        else:
            model.load_state_dict(torch.load(model_path, map_location=config["device"]))
            prompt = input("💬 Введите начальный текст: ")
            result = generate_text(model, tokenizer, prompt, max_new_tokens=50)
            print("\n🧠 Сгенерированный текст:\n")
            print(result)

    else:
        print("⚠️ Неизвестный режим. Введите 'train' или 'gen'.")

