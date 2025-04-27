# 🧠 Abschlussprojekt: Entwicklung eines eigenen Sprachmodells

README.md (English version)
 
# Autoregressive Language Model (Decoder-only Transformer, GPT-style)

This project implements an **autoregressive language model** based on a **decoder-only Transformer** architecture (GPT-style) using **PyTorch**.  
The model was trained on Shakespearean text and is designed for text generation tasks.

## ✨ Key Features

- Based on **PyTorch**.
- Uses **TransformerDecoderLayer** as the core building block.
- Training optimized with **CrossEntropyLoss**.
- Model checkpoints saved at **every epoch**.
- Final model saved separately.
- Simple inference script for text generation.
- Generation supports **temperature** and **top-k sampling**.

## 🏗️ Model Architecture

| Component           | Description                                           |
|---------------------|--------------------------------------------------------|
| Token Embedding     | Embedding of input tokens into vector space            |
| Position Embedding  | Embedding of positional information                   |
| Transformer Blocks  | Stack of multiple `TransformerDecoderLayer` modules   |
| Output Layer        | Linear projection to vocabulary size                  |

The architecture resembles the GPT-style models:  
**Input tokens → Embeddings → Transformer Decoder Blocks → Output logits → Softmax probabilities.**

## ⚙️ Main Hyperparameters

| Parameter         | Meaning                                      |
|-------------------|----------------------------------------------|
| MAX_SEQ_LEN       | Maximum input sequence length (e.g., 128)    |
| d_model           | Dimension of token embeddings               |
| nhead             | Number of attention heads                   |
| num_layers        | Number of TransformerDecoder layers         |
| dim_feedforward   | Size of feed-forward network inside layers  |
| dropout           | Dropout rate inside the model               |
| learning_rate     | Initial learning rate for Adam optimizer    |

## 🧠 Loss Function

The model uses **CrossEntropyLoss**:
- It compares the model's output logits with the ground-truth next tokens.
- Standard choice for language modeling tasks.
- Encourages the model to predict the correct next token given previous tokens.

## 🧪 Training Details

During training:
- **Each epoch**: the model saves a checkpoint (`model_epoch_X.pt`).
- **Final model** is also saved.
- The learning rate decreases after the 8th epoch to stabilize training.
- Evaluation is performed using validation loss (`Val Loss`).

**Example of Training Progress:**

| Epoch | Train Loss | Val Loss | Learning Rate |
|------|------------|----------|---------------|
| 1    | 6.3750     | 5.5461   | 0.000300       |
| 2    | 5.1345     | 4.9006   | 0.000300       |
| 3    | 4.6538     | 4.6558   | 0.000300       |
| 4    | 4.3609     | 4.5320   | 0.000300       |
| 5    | 4.1176     | 4.4622   | 0.000300       |
| ...  | ...        | ...      | ...            |

## 🎯 Inference (Text Generation)

**Sample prompt and generation:**

Prompt:
I have a lot more to learn about these people today.


Generated text:
KING RICHARD III: And I see of all my heart.

STANLEY: Is my hand and his face?

KING RICHARD III: Why be not be the Earl of what news; and for your heart Hath the Earl of thy love in mine, Let'st: ...


Another example:

Prompt:
I swear by the name of the king


Generated text:
The heart of it off from the fire, which Would so to his own love from these thy fortune: 
All that doth they do give; For this time for not have done withal, Let's not like an old women of war, 
The time that I must bring the one of a gentle breath ...


## 📦 Model Files

- Models are saved at each epoch as `model_epoch_X.pt`.
- Final selected model should be uploaded to Hugging Face.

## 🗺️ Model Diagram (Mini-Map)

Here is a high-level map of the model:

Input tokens ↓ Token Embedding ↓ Positional Embedding ↓ TransformerDecoder Layers (stacked) ↓ 
Linear Layer (to vocab size) ↓ Logits (before Softmax)

---

## 📊 Analysis of the Latest Experiments

After analyzing the final training runs:
- **Best performing configuration**:
  - `MAX_SEQ_LEN = 128`
  - `d_model = 384`
  - `nhead = 8`
  - `num_layers = 8`
  - `dim_feedforward = 2048`
  - `dropout = 0.1`
  - `learning_rate = 3e-4`
- Achieved:
  - **Train Loss ≈ 3.7**
  - **Val Loss ≈ 4.38** after 5 epochs.

**Key Observations:**
- Increasing `d_model` from 256 to 384 improved quality.
- Reducing `dropout` to `0.1` gave a more stable convergence.
- Larger models (e.g., `d_model = 512`) struggled to converge due to overfitting and resource constraints.
- Sequence length `128-256` is optimal for this dataset.

## ✅ Recommendations

- Use the model trained with the parameters listed above.
- Optionally fine-tune further on a larger Shakespeare corpus.
- Consider beam search for even higher generation quality if desired.

---

## 🤝 Acknowledgments

- Based on principles described in the original GPT papers.
- Inspired by HuggingFace Transformers library.
- Special thanks to the guidance during experiments and optimization process.

## Hugging Face Model
https://huggingface.co/VadimHammer/my_shakespeare_model
---

# 🚀 Good Luck and Happy Deploying!
