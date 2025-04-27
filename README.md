# Tiny Shakespeare Transformer

A lightweight character-level language model based on the Transformer architecture, trained on the Tiny Shakespeare dataset.

## Overview

This project demonstrates how to build a simple decoder-only Transformer model from scratch using PyTorch.  
It includes full pipeline setup: data loading, tokenization, model training, evaluation, and saving artifacts.

- **Dataset**: Tiny Shakespeare
- **Tokenizer**: Character-level tokenizer based on GPT-2
- **Model**: Custom Transformer Decoder
- **Frameworks**: PyTorch, Hugging Face Transformers, Datasets
- **Logging**: Weights and Biases (wandb)

## Model Architecture

- Token Embedding Layer
- Multiple Transformer Decoder Layers
- GELU Activation Functions
- Dropout Regularization
- Final Linear Layer projecting to vocabulary size

## Hyperparameters

| Parameter        | Value  |
|------------------|--------|
| Batch size       | 64     |
| Block size       | 128    |
| Embedding dim    | 256    |
| Number of heads  | 4      |
| Number of layers | 4      |
| Dropout          | 0.1    |
| Learning rate    | 3e-4   |
| Epochs           | 3–5    |

## Training

- Dataset is tokenized at the character level.
- Inputs are grouped into fixed-length sequences (`block_size`).
- Training and validation split (80/20).
- Model is optimized using the AdamW optimizer.
- Loss is computed using CrossEntropyLoss.

Training metrics (loss, learning rate, epoch time) are logged to [Weights and Biases Project Dashboard](https://wandb.ai/honcharova-de-hannover/LanguageModel_Project?nw=nwuserhoncharovade).

## Model Access

The trained model and tokenizer are available on Hugging Face Hub:

- [NataliaH/tiny_shakespeare_transformer on Hugging Face](https://huggingface.co/NataliaH/tiny_shakespeare_transformer)

## How to Use

1. Install required dependencies:

    ```bash
    pip install torch transformers datasets wandb scikit-learn
    ```

2. Load the model and tokenizer:

    ```python
    import torch
    from transformers import AutoTokenizer

    model = TransformerModel(...)  # Initialize model architecture
    model.load_state_dict(torch.load("model.pth"))
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ```

3. Generate text or fine-tune further as needed.

## Project Links

- [wandb Dashboard](https://wandb.ai/honcharova-de-hannover/LanguageModel_Project?nw=nwuserhoncharovade)
- [Model on Hugging Face](https://huggingface.co/NataliaH/tiny_shakespeare_transformer)

## License

This project is licensed under the MIT License.

---
