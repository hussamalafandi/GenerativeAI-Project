import torch
import torch.nn.functional as F
from model import DecoderLanguageModel
from transformers import GPT2Tokenizer

def generate_text(model, tokenizer, prompt, max_len=50, seq_len=32, top_k=50):
    model.eval()
    input_ids = torch.tensor(tokenizer.encode(prompt))[:seq_len].unsqueeze(0).to("cpu")
    generated = input_ids
    for _ in range(max_len):
        input_seq = generated[:, -seq_len:] if generated.size(1) > seq_len else generated
        with torch.no_grad():
            tgt_mask = model.generate_mask(input_seq.size(1)).to("cpu")
            output = model(input_seq, tgt_mask)
        logits = output[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(top_k_probs, num_samples=1)
        next_token = top_k_indices.gather(-1, next_token)
        generated = torch.cat((generated, next_token), dim=1)
    return tokenizer.decode(generated.squeeze().tolist())

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    model = DecoderLanguageModel(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, max_len=100)
    model.load_state_dict(torch.load("checkpoints/decoder_model_epoch_5.pt"))
    model.to("cpu")
    model.eval()
prompt = "To be or not to be that is the question"
generated_text = generate_text(model, tokenizer, prompt)
print(f"Generated: {generated_text}")

