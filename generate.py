from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("./gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2-finetuned")

prompt = "To be or not to be"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50, top_k=50, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

