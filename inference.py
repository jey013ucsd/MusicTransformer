import os
import json
import torch
import torch.nn.functional as F
from models.MusicTransformerv2 import MusicTransformerv2
from data_processing.decoding import decode_to_midi_basic_vocab, decode_to_midi_basic_vocab_velocity_bins, decode_to_midi_multi_instr_vocab
import mido
from mido import MidiFile, MidiTrack, Message, second2tick

# Paths
EXPERIMENT_NAME = "v2_100epoch_half_dataset"
OUTPUT_NAME = "paris_test_t0.85_tk355"
VOCAB_PATH = "models/vocab/multi_instr_vocab.json"
CHECKPOINT_PATH = f"{EXPERIMENT_NAME}/checkpoints/model_epoch_100.pt"
#CHECKPOINT_PATH = f"{EXPERIMENT_NAME}/model_final.pt"


N_EMBD = 1024
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0
MAX_SEQ_LENGTH = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GENERATED_TOKENS = 2048  # maximum number of tokens to generate
TEMPERATURE = 0.85
TOP_K = 355



with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

idx_to_token = {int(v): k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")


print(f"Device: {DEVICE}")

model = MusicTransformerv2(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
    max_len=MAX_SEQ_LENGTH
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state_dict"])
model.eval()
print(f"Loaded model from {CHECKPOINT_PATH}")



def generate_sequence(model, seed_tokens, max_tokens=MAX_GENERATED_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
    """
    Autoregressively generate a sequence of token ids starting from seed_tokens.

    """
    model.eval()
    generated = seed_tokens.copy()
    for _ in range(max_tokens):
        input_seq = torch.tensor(generated[-MAX_SEQ_LENGTH:], dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(input_seq)
        logits = logits[0, -1, :] / temperature

        if top_k > 0:
            topk_logits, topk_indices = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits)
            probs[topk_indices] = F.softmax(topk_logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        
        if idx_to_token.get(next_token, "") == "TOKEN_END":
            break
    return generated


#input sequence
seed_prompt = [53, 380, 65, 380, 280, 181, 193, 261, 60, 380, 273, 188, 292, 68, 380, 56, 380, 280, 196, 184, 261, 60, 380, 273, 188, 298, 55, 380, 67, 380, 280, 183, 195, 261, 60, 380, 273, 188, 292, 53, 380, 65, 380, 280, 181, 193, 261, 60, 380, 273, 188, 261, 72, 380, 84, 380, 60, 380, 273, 200, 212, 188, 292, 53, 380, 65, 380, 280, 181, 193, 261, 60, 380, 273, 188, 292, 68, 380, 56, 380, 280, 196, 184, 261, 60, 380, 273, 188, 298, 55, 380, 67, 380, 280, 183, 195, 261, 60, 380, 273, 188, 292, 65, 380, 53, 380, 280, 193, 181, 261, 60, 380, 273, 188, 261, 72, 380, 84, 380, 60, 380, 273, 200, 212, 188]






generated_sequence = generate_sequence(model, seed_prompt)
generated_tokens = [idx_to_token[token_id] for token_id in generated_sequence]

print("Generated Sequence:")
print(" ".join(generated_tokens))


tokens = generated_tokens
decode_to_midi_multi_instr_vocab(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}cheated.mid", turn_off_notes=True, max_len=350)
#decode_to_midi_basic_vocab_velocity_bins(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}.mid", turn_off_notes=False)