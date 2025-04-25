# Decoder-Only Transformer Small

This is a minimal autoregressive transformer decoder model trained on a small toy dataset using PyTorch.

## Details

 Parameters: < 1M  
- Architecture: Decoder-only (GPT-style)  
- Layers: 1  
- Hidden Size: 64  
- Attention Heads: 4  
- Trained on: synthetic text  

**Model:** Decoder-only Transformer  
**Problem type:** Text generation  
**Model depth:** 1 decoder layer  
**Hidden size:** 64  
**Attention heads:** 4  
**Vocabulary size:** 50257  
**Max sequence length:** 64  

## Usage
You can use this model to generate text using the `transformers` library from Hugging Face:

The code uses the Tiny Shakespeare dataset, which is downloaded from GitHub (https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
python 

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "hannanechiporenko25/decoder-only-transformer-small"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "To be or not to be"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(inputs['input_ids'], max_length=50)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

[wandb.ai](https://wandb.ai/annadesignerart22-uni/transformer_decoder_only_final?nw=nwuserannadesignerart22)— look at the graphs `train_loss` and `val_loss`
[huggingface](https://huggingface.co/hannanechiporenko25/decoder-only-transformer-small/blob/main/README.md)— look at the card readme.md