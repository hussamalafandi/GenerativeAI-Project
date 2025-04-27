import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb
from tqdm import tqdm

# ======================== Config ========================
config = {
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "model_dim": 256,
    "n_heads": 4,
    "n_layers": 4,
    "block_size": 64,
    "dataset": "wikitext",
    "dataset_config": "wikitext-2-raw-v1"
}

# ======================== Device ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================== wandb ========================
wandb.init(project="my-transformer-lm", config=config)

# ======================== Tokenizer & Dataset ========================
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config["vocab_size"] = tokenizer.vocab_size

raw_dataset = load_dataset(config["dataset"], config["dataset_config"])

class TokenDataset(Dataset):
    def __init__(self, texts, block_size):
        self.data = []
        for txt in texts:
            # токенизируем и сразу добавляем паддинг до block_size
            tokenized = tokenizer.encode(
                txt['text'], 
                truncation=True, 
                max_length=block_size, 
                padding="max_length"
            )
            # теперь все последовательности длины block_size
            self.data.append(torch.tensor(tokenized))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_texts = raw_dataset["train"]
val_texts = raw_dataset["validation"]
train_dataset = TokenDataset(train_texts, config["block_size"])
val_dataset = TokenDataset(val_texts, config["block_size"])

train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], drop_last=True)

# ======================== Model ========================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, block_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, block_size, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]
        x = x.transpose(0, 1)  # Transformer expects seq_len, batch, embed
        x = self.encoder(x)
        x = x.transpose(0, 1)  # Back to batch, seq_len, embed
        return self.fc(x)

model = TransformerLM(
    vocab_size=config["vocab_size"],
    embed_dim=config["model_dim"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"],
    block_size=config["block_size"]
).to(device)

# ======================== Train ========================
def train(model, train_dataloader, val_dataloader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training"):
            inputs = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)

            logits = model(inputs)
            logits = logits.view(-1, config["vocab_size"])
            targets = targets.reshape(-1)

            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} - Validation"):
                inputs = batch[:, :-1].to(device)
                targets = batch[:, 1:].to(device)

                logits = model(inputs)
                logits = logits.view(-1, config["vocab_size"])
                targets = targets.reshape(-1)

                loss = loss_fn(logits, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch+1})

train(model, train_dataloader, val_dataloader, config["epochs"])
wandb.finish()

# ======================== Generate ========================
def generate_text(
    model, tokenizer, prompt, max_length=50, device=device,
    temperature=1.0, top_k=None, top_p=None
):
    # Check for simultaneous use top_k и top_p
    if top_k is not None and top_p is not None:
        raise ValueError("cannot be used simultaneously top_k и top_p. Choose one.")

    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids

    for _ in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                probs = F.softmax(top_k_values, dim=-1)
                next_token = top_k_indices[0, torch.multinomial(probs, num_samples=1)]
            # Top-p (nucleus) sampling
            elif top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_keep = cumulative_probs <= top_p
                sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
                sorted_indices_to_keep[..., 0] = True

                indices_to_keep = sorted_indices[sorted_indices_to_keep]
                filtered_logits = logits[:, indices_to_keep]
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = indices_to_keep[torch.multinomial(probs, num_samples=1)]
            else:
                # Greedy (argmax) decoding
                probs = F.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)

            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)