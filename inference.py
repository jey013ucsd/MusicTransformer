import os
import json
import torch
import torch.nn.functional as F
from models.MusicTransformer import MusicTransformer
from models.MusicTransformerv2 import MusicTransformerv2
from data_processing.decoding import decode_to_midi_basic_vocab, decode_to_midi_basic_vocab_velocity_bins, decode_to_midi_multi_instr_vocab
import mido
from mido import MidiFile, MidiTrack, Message, second2tick
import yaml
def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

# Paths
EXPERIMENT_NAME = config['Training']['EXPERIMENT_NAME']
VOCAB = config['VOCAB']
OUTPUT_NAME = config['Inference']['OUTPUT_NAME']
CHECKPOINT_PATH = f"{EXPERIMENT_NAME}/{config['Inference']['MODEL_NAME']}"
#CHECKPOINT_PATH = f"{EXPERIMENT_NAME}/model_final.pt"

N_EMBD = config['MODEL']['N_EMBD']
N_HEAD = config['MODEL']['N_HEAD']
N_LAYER = config['MODEL']['N_LAYER']
DROPOUT = 0
MAX_SEQ_LENGTH = config['MODEL']['BLOCK_SIZE']


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GENERATED_TOKENS = config['Inference']['MAX_GENERATED_TOKENS']  # maximum number of tokens to generate
TEMPERATURE = config['Inference']['TEMPERATURE']
TOP_K = config['Inference']['TOP_K']

if VOCAB == 'BASIC' or 'BASIC_VELOCITY_BINS':
    VOCAB_PATH = "models/vocab/basic_vocab.json"
if VOCAB == 'MULTI_STR':
    VOCAB_PATH = "models/vocab/multi_instr_vocab.json"

with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

idx_to_token = {int(v): k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")


print(f"Device: {DEVICE}")

if config['MODEL']['MODEL'] == 'V2':
    model = MusicTransformerv2(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
        max_len=MAX_SEQ_LENGTH
    ).to(DEVICE)
elif config['MODEL']['MODEL'] == 'V1':
    model = MusicTransformer(
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
seed_prompt = config['Inference']['SEED_PROMPT']


generated_sequence = generate_sequence(model, seed_prompt)
generated_tokens = [idx_to_token[token_id] for token_id in generated_sequence]

print("Generated Sequence:")
print(" ".join(generated_tokens))


tokens = generated_tokens
#decode_to_midi_multi_instr_vocab(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}cheated.mid", turn_off_notes=True, max_len=350)
if VOCAB == 'BASIC':
    decode_to_midi_basic_vocab(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}.mid", turn_off_notes=False)
elif VOCAB == 'BASIC_VELOCITY_BINS':
    decode_to_midi_basic_vocab_velocity_bins(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}.mid", turn_off_notes=False)
elif VOCAB == 'MULTI_STR':
    decode_to_midi_multi_instr_vocab(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}.mid", turn_off_notes=False)