import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from torch.utils.data import DataLoader
from model import DecoderLanguageModel
from dataset import prepare_data  # Korrigierter Import
import math

def train_model(train_dataset, val_dataset, tokenizer, epochs=5, batch_size=4, checkpoint_dir="checkpoints"):
    print("Starting training...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Creating dataloaders with batch_size={batch_size}...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print(f"Train loader size: {len(train_loader)}, Val loader size: {len(val_loader)}")
    vocab_size = tokenizer.vocab_size
    model = DecoderLanguageModel(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, max_len=100)
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    print("Initializing wandb...")
    wandb.init(project="decoder-language-model", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "d_model": 128,
        "num_layers": 2,
        "nhead": 4,
        "seq_len": 32
    })
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_perplexity = math.exp(avg_train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src, tgt = batch
                src, tgt = src.to(device), tgt.to(device)
                output = model(src)
                val_loss += criterion(output.view(-1, vocab_size), tgt.view(-1)).item()
        avg_val_loss = val_loss / len(val_loader)
        val_perplexity = math.exp(avg_val_loss)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_perplexity": train_perplexity,
            "val_loss": avg_val_loss,
            "val_perplexity": val_perplexity
        })
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}")
        checkpoint_path = os.path.join(checkpoint_dir, f"decoder_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    return model, tokenizer

if __name__ == "__main__":
    print("Preparing data...")
    train_dataset, val_dataset, tokenizer = prepare_data("data/input.txt", "data/tiny_shakespeare.h5")
    print("Starting training...")
    model, tokenizer = train_model(train_dataset, val_dataset, tokenizer)