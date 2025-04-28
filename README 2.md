
# 🚀 Fine-Tuning Tiny GPT-2 on Shakespeare Text

This project demonstrates how to fine-tune a small GPT-2 model (less than 1M parameters) on the **Tiny Shakespeare** dataset using PyTorch and Hugging Face `transformers`, with logging through Weights & Biases (wandb).

---

## 📦 Install Dependencies

```bash
pip install -U transformers datasets wandb huggingface_hub
pip install fsspec==2023.3.2
```

*(Optional for GPU)*

```bash
pip install nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-cupti-cu12==12.4.127 nvidia-cuda-runtime-cu12==12.4.127
```

---

## 🔑 Authentication

```python
from huggingface_hub import login
import wandb

login(token="YOUR_HF_TOKEN")
wandb.login(key="YOUR_WANDB_API_KEY")
```

> Replace `YOUR_HF_TOKEN` and `YOUR_WANDB_API_KEY` with your own tokens.

---

## 📚 Prepare the Dataset

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O tinyshakespeare.txt
```

```python
from datasets import load_dataset

dataset = load_dataset('text', data_files={'train': 'tinyshakespeare.txt'})
dataset["train"] = dataset["train"].select(range(3000))  # Using first 3000 examples
```

---

## ✂️ Tokenization

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_special_tokens_mask=True,
        return_attention_mask=True,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```

---

## 🏋️ Fine-Tuning the Model

```python
from transformers import GPT2Config, GPT2LMHeadModel, TrainingArguments, Trainer

config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=128,
    n_ctx=128,
    n_embd=128,
    n_layer=4,
    n_head=2,
    pad_token_id=tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(config)

training_args = TrainingArguments(
    output_dir="./tiny_gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    logging_steps=500,
    learning_rate=5e-4,
    weight_decay=0.01,
    report_to="wandb",
    push_to_hub=True,
    hub_model_id="hannanechiporenko25/tiny-gpt2-shakespeare",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.push_to_hub()
tokenizer.push_to_hub("hannanechiporenko25/tiny-gpt2-shakespeare")
```

---

## ✨ Generate Text After Fine-Tuning

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("hannanechiporenko25/tiny-gpt2-shakespeare")
tokenizer = GPT2Tokenizer.from_pretrained("hannanechiporenko25/tiny-gpt2-shakespeare")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

start_text = "to be or not to be"
input_ids = tokenizer.encode(start_text, return_tensors="pt").to(device)

output_ids = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    pad_token_id=tokenizer.pad_token_id,
)

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Text:\n", generated_text)
```

---

## 📈 Monitoring Training
- All metrics like `train_loss` and `val_loss` are automatically logged to [Weights & Biases](https://wandb.ai/).
- Model checkpoints are pushed to [Hugging Face Hub](https://huggingface.co/).

---

# ✅ Project Complete!

---

## 📌 Project Structure

```
tiny_gpt2/
├── tinyshakespeare.txt
├── training_script.py
├── README.md
├── logs/
└── LICENSE
``
[wandb.ai]https://wandb.ai/annadesignerart22-uni/huggingface?nw=nwuserannadesignerart22 — look at the graphs `train_loss` `
