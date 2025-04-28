# 🤖 GPT-2 Shakespeare Finetuned

Dies ist ein kleines, finetuned GPT-2 Modell, das auf dem [Tiny Shakespeare Dataset](https://huggingface.co/datasets/karpathy/tiny_shakespeare) trainiert wurde.  
Es nutzt die `Trainer`-API von Hugging Face und ist ideal für **textgenerierende Experimente im Shakespeare-Stil** 📝🎭

## 🔧 Training Details

- Modell: `gpt2`
- Datensatz: `karpathy/tiny_shakespeare`
- Tokenizer: `GPT2Tokenizer` (mit `eos_token` als Padding-Token)
- Framework: 🤗 `transformers` + `Trainer`
- Logging: 📊 [`Weights & Biases`](https://wandb.ai/basan-1994-15-hochschule-hannover/huggingface?nw=nwuserbasan199415)
- Epochs: 3
- Batch Size: 2 (pro Gerät, wie in `per_device_train_batch_size`)
- Sequence Length: 512 (festgelegt durch `block_size`)
- Loss: Causal Language Modeling (Auto-regressive, kein Masked LM)


## 📦 Installation

```bash
pip install transformers datasets wandb
