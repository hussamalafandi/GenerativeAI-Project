# Link zu Hugging Face
https://huggingface.co/WenWebProjekt/mini-decoder/tree/main

# Link zu wandb
https://wandb.ai/wenwenhandy-hochschule-hannover/tiny-transformer/runs/iyggw8c7/workspace?nw=nwuserwenwenhandy


# Mini Decoder GPT

This is a simple language model (Decoder-only) built with PyTorch. The training data is "Tiny Shakespeare", using the GPT2 tokenizer.

## Model Structure

- The model structure is based on TransformerDecoder.
- The total number of parameters is less than 1 million and can be trained on CPU.

## Training loop

- Training 5 Epoche
- Use **wandb** to record the training process.

## Model download

You can download this model for testing or fine-tuning on Hugging Face.

## use it

```bash
# install
pip install transformers datasets torch wandb

# loading
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("WenWebProjekt/mini-decoder")
tokenizer = AutoTokenizer.from_pretrained("WenWebProjekt/mini-decoder")

# Example
input_text = "To be, or not to be"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(inputs["input_ids"])
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
