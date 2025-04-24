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
    "block_size": 256,
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

train_ids = tokenizer(train_text, return_tensors="pt")["input_ids"].squeeze()[:20000]
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
    def __init__(self, vocab_size, d_model=96, n_heads=2, n_layers=3):
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
def generate_text(model, tokenizer, prompt, max_new_tokens=30, temperature=0.9,     ):
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

# 🧠 Продвинутая генерация текста
def generate_text_advanced(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50, top_p=1.0, num_return_sequences=1):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(config["device"])
    
    generated_outputs = []

    for _ in range(num_return_sequences):
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(generated)
                logits = outputs[:, -1, :] / temperature

                # Top-k
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(logits, top_k)
                    probs = torch.zeros_like(logits).scatter(1, top_k_indices, top_k_values)
                else:
                    probs = logits

                # Softmax + Top-p
                sorted_logits, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                if top_p < 1.0:
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    sorted_logits[sorted_indices_to_remove] = -float("Inf")

                probs = torch.softmax(sorted_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                next_token_unsorted = sorted_indices.gather(-1, next_token)
                generated = torch.cat([generated, next_token_unsorted], dim=1)

        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        generated_outputs.append(text)

    return generated_outputs


# 🚦 Основной запуск
if __name__ == "__main__":
    mode = input("\n🤖 Выбери режим: [1] Обучение | [2] Генерация | [3] Продвинутая генерация: ")

    model = TinyDecoder(vocab_size=len(tokenizer)).to(config["device"])
    model_path = "tiny_decoder.pt"

    if mode == "1":
        print("🚀 Запуск обучения...")
        wandb.init(project="tiny-language-model", config=config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
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

    elif mode == "2":
        prompt = input("Введите начальный текст: ")
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=50)
        print("\n🧠 Generated text:\n", generated)

    elif mode == "3":
        prompt = input("✏️ Введите начальный текст: ")
        temperature = float(input("🔥 Temperature (напр. 1.0): ") or 1.0)
        top_k = int(input("🎯 Top-k (напр. 50): ") or 50)
        top_p = float(input("🔮 Top-p (напр. 0.9): ") or 1.0)
        num_seq = int(input("📎 Сколько сгенерировать вариантов? (по умолч. 1): ") or 1)

        outputs = generate_text_advanced(
            model, tokenizer, prompt,
            max_new_tokens=60,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_seq
        )

        for i, out in enumerate(outputs):
            print(f"\n🧠 Generated #{i+1}:\n{out}")

    else:
        print("⚠️ Неизвестный режим. Завершение.")
