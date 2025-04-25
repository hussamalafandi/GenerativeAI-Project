# 🧠 Abschlussprojekt: Entwicklung eines eigenen Sprachmodells

README.md (English version)
 
# 🧠 Mini Language Model in PyTorch

## 📌 Overview

This project implements a simple **autoregressive language model** (decoder-only Transformer, GPT-style) using **PyTorch**.

Features:
- Training on the **Tiny Shakespeare** dataset
- Text generation from a user-provided prompt
- Adjustable **temperature** and **top-k** sampling
- Saves models per epoch into the `checkpoints/` folder
- Logging with **Weights & Biases** (`wandb`)
- Lets you choose a specific model version for generation

---

## 🧱 Project Structure

- `main.py`: the main file — includes
  - model definition,
  - training function (`train()`),
  - generation function (`generate()`),
  - interactive menu for mode selection
- `checkpoints/`: automatically created, contains per-epoch models (`model_epoch_N.pt`)
- `my_model.pt`: final model version

---

## 📦 Install dependencies

 
pip install -r requirements.txt
requirements.txt:
 
torch
transformers
wandb
huggingface_hub
requests

🚀 Run the project
bash
КопироватьРедактировать
python main.py
You’ll be prompted to choose a mode:
 
Select mode:
1 — Train the model
2 — Generate text

⚙️ Training
The model is trained on text chunks of MAX_SEQ_LEN = 128.
Each epoch saves a checkpoint into checkpoints/model_epoch_N.pt.
The following are logged:
    • train_loss
    • val_loss
    • learning_rate
Example output:
 
Epoch 5 | Train Loss: 4.25 | Val Loss: 4.67 | LR: 0.000300
📦 Model saved: checkpoints/model_epoch_5.pt
