import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2Tokenizer
from model import DecoderLanguageModel

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, temperature=1.2):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    if top_p > 0.0:
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = -float('Inf')

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')

    return logits


def generate_text_advanced(model, tokenizer, prompt, max_len=50, seq_len=32, top_k=50, top_p=0.9, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor(tokenizer.encode(prompt))[:seq_len].unsqueeze(0).to(device)
    generated = input_ids

    for _ in tqdm(range(max_len), desc="Generating"):
        input_seq = generated[:, -seq_len:]
        with torch.no_grad():
            tgt_mask = model.generate_mask(input_seq.size(1)).to(device)
            output = model(input_seq, tgt_mask)

        logits = output[:, -1, :]  # letzte Token-Logits
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, temperature=temperature)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1)

        # Optional: Stoppen wenn <eos> generiert
        if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated.squeeze().tolist())
if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    model = DecoderLanguageModel(vocab_size=vocab_size, d_model=128, nhead=4, num_layers=2, max_len=100)
    model.load_state_dict(torch.load("checkpoints/decoder_model_epoch_5.pt"))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    prompt = "To be or not to be"
    generated_text = generate_text_advanced(model, tokenizer, prompt, max_len=50, top_k=40, top_p=0.92, temperature=1.0)
    print("Generated:", generated_text)
