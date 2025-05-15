# %%


from dotenv import load_dotenv
load_dotenv()
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import wandb

# Einstellungen
vocab_size = 50257  # GPT2 vocab
d_model = 64
nhead = 4
dim_feedforward = 128
num_layers = 1
max_seq_len = 64
batch_size = 4
seq_len = 8
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="transformer_decoder_only_final")

# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  #
##
import requests

# Laden Sie die Datei herunter und speichern Sie sie
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

# Sie können die Länge überprüfen
print(f"Wie lange: {len(text)} zeichen")
print(text[:100])  # erst 100 symbol
##
tokens = tokenizer.encode(text)

# Dataset
##
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data = []
        for i in range(0, len(token_ids) - seq_len - 1, seq_len):  # <-- шаг = seq_len
#                for i in range(len(token_ids) - seq_len):
            self.data.append(torch.tensor(token_ids[i:i + seq_len + 1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = TextDataset(tokens, seq_len=64)
 #

#dataset = TextDataset(tokens, seq_len)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset, batch_size=batch_size)
print(f"Num batches per epoch: {len(train_loader)}")

# Model
class DecoderOnlyModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand_as(x)
        x_embed = self.embedding(x) + self.pos_embedding(pos)

        # Erstellen einer Maske (causal)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        x_out = self.decoder(x_embed.transpose(0, 1), torch.zeros_like(x_embed.transpose(0, 1)), tgt_mask=tgt_mask)
        logits = self.output(x_out.transpose(0, 1))
        return logits

model = DecoderOnlyModel(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_seq_len).to(device)

# Training.... Bewertung anhand eines Validierungssatzes (validate): viziv v konce epohi  in train()
def validate(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train(model, train_loader, val_loader, epochs, lr=1e-4,log_evary=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    global_step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step+=1

 # Wir protokollieren nicht jeden Schritt, sondern jeden log_every log_every
            if global_step % log_evary == 0:
                wandb.log({"step_train_loss": loss.item(), "step": global_step})
       # Validation
        val_loss = validate(model, val_loader, loss_fn)
        avg_train_loss = total_loss / len(train_loader)
                                      #vizov proverka
       # wandb.log({"epoch": epoch+1, "train_loss": total_loss / len(train_loader), "val_loss": val_loss},step=epoch)
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "step": global_step  
        })

  #      print(f"Epoch {epoch+1}: Train Loss = {total_loss / len(train_loader):.4f}, Val Loss = {val_loss:.4f}")
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
         # wandb.define_metric("epoch")
        # wandb.define_metric("train_loss", step_metric="epoch")
        # wandb.define_metric("val_loss", step_metric="epoch")

train(model, train_loader, val_loader, epochs)

##
# ====== Function generative text ======
def generate_text(model, tokenizer, start_text, max_len=128, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)
    for _ in range(max_len):
         # Schneiden input_ids, falls mehr max_seq_len
         if input_ids.shape[1] >= model.pos_embedding.num_embeddings:  
            input_ids = input_ids[:, -model.pos_embedding.num_embeddings:]

#         if input_ids.shape[1] >= max_seq_len:
#             input_ids = input_ids[:, -max_seq_len:]
         seq_len = input_ids.size(1)
         pos = torch.arange(0, seq_len, device=device).unsqueeze(0)
         x_embed = model.embedding(input_ids) + model.pos_embedding(pos)

         tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

          # Leehr kontext — decoder-only model
         memory = torch.zeros_like(x_embed).to(device)
         out = model.decoder(x_embed.transpose(0, 1), memory.transpose(0, 1), tgt_mask=tgt_mask)
         logits = model.output(out.transpose(0, 1))

         next_token_logits = logits[:, -1, :] / temperature
         probs = torch.softmax(next_token_logits, dim=-1)
         next_token = torch.multinomial(probs, num_samples=1)
         input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# ====== Generated und Login dursch wandb ======

# start_text = "Shall I compare thee to a summer's day?"
# generated = generate_text(model, tokenizer, start_text, max_len=100)
start_text = "To be or not to be"
generated_text = generate_text(model, tokenizer, start_text, max_len=50, temperature=0.8)

print("\n===== Generated Text =====")
print(generated_text)

# login Generated Text in wandb
wandb.log({"generated_text": generated_text})

print(device)
if torch.cuda.is_available():
    print(torch.cuda.memory_summary())
else:
    print("CUDA kein — nutzen CPU")

##print(device)
##print(torch.cuda.memory_summary())
#
# Einstellen schlussel
from huggingface_hub import login

hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token is None:
    raise RuntimeError("HUGGINGFACE_TOKEN kein в .env")
login(token=hf_token)



#from transformers import AutoModel, AutoTokenizer

#model_name = "bert-base-uncased"

# Herunten und tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
#model = AutoModel.from_pretrained(model_name, use_auth_token=True)

######   Speichern model und AutoTokenizer
from transformers import AutoTokenizer
####from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub import HfApi, create_repo, upload_folder
import torch
import os

model_name = "decoder-only-transformer-small"

# Neu folder fur model
save_dir = f"./{model_name}"
os.makedirs(save_dir, exist_ok=True)

# Speichern modell
torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

# Speichern configuration
with open(os.path.join(save_dir, "config.json"), "w") as f:
    f.write("""{
        "model_type": "decoder-only",
        "vocab_size": 50257,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 1,
        "max_position_embeddings": 64
    }""")

# Speichern Tokinizator
tokenizer.save_pretrained(save_dir)

###  Herunten laden in Hugging Face
from huggingface_hub import create_repo, upload_folder

repo_id = "invest-ua1/decoder-only-transformer-small"
#repo_id = "invest-ua1"
#create_repo(repo_id, private=False)

####
# Versuchen wir, ein Repository zu erstellen. Wenn es bereits vorhanden ist, ignorieren Sie es.
api = HfApi()

try:
    create_repo(repo_id, private=False)  # Erstellen Sie ein Repository, falls es nicht vorhanden ist
except Exception as e:
    print(f"Repository already exists or error occurred: {e}")

# Ordner mit Modell wird geladen
upload_folder(
    folder_path=save_dir,
    repo_id=repo_id,
    commit_message="Upload small decoder-only model"
)

print(f"Model successfully uploaded to Hugging Face at {repo_id}")
####




