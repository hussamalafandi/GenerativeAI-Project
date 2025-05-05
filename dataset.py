import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        return (self.tokens[idx:idx+self.seq_len], 
                self.tokens[idx+1:idx+self.seq_len+1])

def prepare_data(data_path, h5_path, seq_len=32, val_split=0.1):
    # Lade Text
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()[:500000]  # Begrenze auf 500k Zeichen

    # Tokenisierung
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = np.array(tokenizer.encode(text))

    # Speichere tokens in HDF5
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("tokens", data=tokens)

    # Train/Val-Split
    train_size = int((1 - val_split) * len(tokens))
    train_tokens = tokens[:train_size]
    val_tokens = tokens[train_size:]

    # Erstelle Datasets
    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)

    return train_dataset, val_dataset, tokenizer