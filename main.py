# 📦 Импорты 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import wandb
import requests
import gc

# ========== Глобальные параметры ==========
# MAX_SEQ_LEN — максимальная длина, которую модель может обработать
# BLOCK_SIZE — размер входа для обучения: MAX_SEQ_LEN + 1
MAX_SEQ_LEN = 128
BLOCK_SIZE = MAX_SEQ_LEN + 1

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# ========== 1. Архитектура модели ==========
class MyDecoderModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, dim_feedforward=512, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation='gelu')
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device) * float('-inf'), diagonal=1)
        tgt_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        # Создаём фиктивный memory (нулевой, той же формы)
        memory = torch.zeros_like(x).transpose(0, 1)

        x = self.decoder(
            tgt=x.transpose(0, 1),
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        x = x.transpose(0, 1)
        return self.output_projection(x)

# ========== 2. Dataset и загрузка ==========
class TextDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        x = self.input_ids[idx][:-1]
        y = self.input_ids[idx][1:]
        return x, y

def chunk_dataset(input_ids, block_size=129):
    num_chunks = len(input_ids) // block_size
    input_ids = input_ids[:num_chunks * block_size]
    input_ids = input_ids.view(num_chunks, block_size)
    return input_ids

def load_dataset(tokenizer, block_size=129):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = requests.get(url).text
    encodings = tokenizer(text, return_tensors="pt", truncation=False, padding=False)
    input_ids = encodings["input_ids"].squeeze()
    chunks = chunk_dataset(input_ids, block_size)
    dataset = TextDataset(chunks)
    return dataset

# ========== 3. Валидация ==========
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

# ========== 4. Обучение ==========
def train():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(tokenizer)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MyDecoderModel(vocab_size=tokenizer.vocab_size, max_seq_len=MAX_SEQ_LEN).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Добавлен адаптивный scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    wandb.init(project="my-language-model")
    epochs = 15

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = validate(model, val_loader, criterion, device)

        # Обновление scheduler'а
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": current_lr
        })
        # Сохраняем модель для каждой эпохи
        # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
        # print(f"📦 Модель сохранена: model_epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), "my_model.pt")
    print("✅ Модель сохранена в my_model.pt")

    clear_gpu_memory()


# ========== 5. Генерация ==========
def sample_logits(logits, temperature=1.0, top_k=0):
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_threshold = values[:, -1].unsqueeze(-1)
        logits[logits < min_threshold] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def generate():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 Новый ввод: выбор модели
    model_path = input("Укажите путь к файлу модели (Enter для 'my_model.pt'): ").strip()
    if model_path == "":
        model_path = "my_model.pt"

    # Загружаем модель
    model = MyDecoderModel(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Параметры генерации
    prompt = input("Введите начальный текст: ")
    temperature = float(input("Температура (например 1.0): ") or "1.0")
    top_k = int(input("Top-k (0 = без ограничения): ") or "0")
    max_new_tokens = int(input("Сколько токенов сгенерировать? ") or "50")

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = sample_logits(next_token_logits, temperature=temperature, top_k=top_k)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    result = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print("\nСгенерированный текст:\n", result)

    # Очистка памяти
    clear_gpu_memory()

# ========== 6. Главное меню ==========
def main():
    print("Выберите режим:")
    print("1 — Обучение модели")
    print("2 — Генерация текста")
    choice = input("Ваш выбор (1/2): ")
    if choice == "1":
        train()
    elif choice == "2":
        generate()
    else:
        print("Неверный выбор")

if __name__ == "__main__":
    main()