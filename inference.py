import os
import json
import torch
import torch.nn.functional as F
from models.MusicTransformer import MusicTransformer
from data_processing.decoding import decode_to_midi_basic_vocab
import mido
from mido import MidiFile, MidiTrack, Message, second2tick

# Paths
EXPERIMENT_NAME = "150epoch_half_dataset"
OUTPUT_NAME = "test"
VOCAB_PATH = "models/vocab/basic_vocab.json"
CHECKPOINT_PATH = f"{EXPERIMENT_NAME}/checkpoints/model_epoch_45.pt"
#CHECKPOINT_PATH = f"{EXPERIMENT_NAME}/model_final.pt"


N_EMBD = 1024
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0
MAX_SEQ_LENGTH = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GENERATED_TOKENS = 1024  # maximum number of tokens to generate
TEMPERATURE = 1.0
TOP_K = 0



with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

idx_to_token = {int(v): k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")


print(f"Device: {DEVICE}")

model = MusicTransformer(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
    max_len=MAX_SEQ_LENGTH
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state_dict"]
                      )
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
seed_prompt = [
    # First Motif: Ascending triplet pattern
    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],  # C4
    vocab["TIME_SHIFT_200ms"],
    vocab["NOTE_ON_64"], vocab["VELOCITY_20"],  # E4
    vocab["TIME_SHIFT_200ms"],
    vocab["NOTE_ON_67"], vocab["VELOCITY_20"],  # G4
    vocab["TIME_SHIFT_200ms"],
    vocab["NOTE_OFF_60"],
    vocab["NOTE_OFF_64"],
    vocab["NOTE_OFF_67"],

    # Descending syncopated answer
    vocab["NOTE_ON_65"], vocab["VELOCITY_20"],  # F4
    vocab["TIME_SHIFT_300ms"],
    vocab["NOTE_OFF_65"],
    vocab["NOTE_ON_62"], vocab["VELOCITY_20"],  # D4
    vocab["TIME_SHIFT_150ms"],
    vocab["NOTE_OFF_62"],
    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],  # C4
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_60"],

    # Repeat motif with variation (higher and faster)
    vocab["NOTE_ON_67"], vocab["VELOCITY_25"],  # G4
    vocab["TIME_SHIFT_180ms"],
    vocab["NOTE_ON_70"], vocab["VELOCITY_25"],  # Bb4
    vocab["TIME_SHIFT_180ms"],
    vocab["NOTE_ON_74"], vocab["VELOCITY_25"],  # D5
    vocab["TIME_SHIFT_180ms"],
    vocab["NOTE_OFF_67"],
    vocab["NOTE_OFF_70"],
    vocab["NOTE_OFF_74"],

    # First Motif: Ascending triplet pattern
    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],  # C4
    vocab["TIME_SHIFT_200ms"],
    vocab["NOTE_ON_64"], vocab["VELOCITY_20"],  # E4
    vocab["TIME_SHIFT_200ms"],
    vocab["NOTE_ON_67"], vocab["VELOCITY_20"],  # G4
    vocab["TIME_SHIFT_200ms"],
    vocab["NOTE_OFF_60"],
    vocab["NOTE_OFF_64"],
    vocab["NOTE_OFF_67"],

    # Descending syncopated answer
    vocab["NOTE_ON_65"], vocab["VELOCITY_20"],  # F4
    vocab["TIME_SHIFT_300ms"],
    vocab["NOTE_OFF_65"],
    vocab["NOTE_ON_62"], vocab["VELOCITY_20"],  # D4
    vocab["TIME_SHIFT_150ms"],
    vocab["NOTE_OFF_62"],
    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],  # C4
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_60"],

    # Repeat motif with variation (higher and faster)
    vocab["NOTE_ON_67"], vocab["VELOCITY_25"],  # G4
    vocab["TIME_SHIFT_180ms"],
    vocab["NOTE_ON_70"], vocab["VELOCITY_25"],  # Bb4
    vocab["TIME_SHIFT_180ms"],
    vocab["NOTE_ON_74"], vocab["VELOCITY_25"],  # D5
    vocab["TIME_SHIFT_180ms"],
    vocab["NOTE_OFF_67"],
    vocab["NOTE_OFF_70"],
    vocab["NOTE_OFF_74"]
]

'''
seed_prompt = [

    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_60"],

    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_60"],

    vocab["NOTE_ON_40"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_40"],

    vocab["NOTE_ON_40"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_40"],
    
    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_60"],

    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_60"],

    vocab["NOTE_ON_40"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_40"],

    vocab["NOTE_ON_40"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_40"],

    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_60"],

    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_60"],

    vocab["NOTE_ON_40"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_40"],

    vocab["NOTE_ON_40"], vocab["VELOCITY_20"],
    vocab["TIME_SHIFT_250ms"],
    vocab["NOTE_OFF_40"],
]
'''
seed_prompt = [vocab["NOTE_ON_72"], vocab["VELOCITY_26"], vocab["NOTE_ON_75"], vocab["VELOCITY_26"], vocab["NOTE_ON_32"], vocab["VELOCITY_25"], vocab["NOTE_ON_60"], vocab["VELOCITY_26"], vocab["NOTE_ON_63"], vocab["VELOCITY_29"], vocab["NOTE_ON_35"], vocab["VELOCITY_29"], vocab["NOTE_ON_49"], vocab["VELOCITY_13"], vocab["NOTE_ON_40"], vocab["VELOCITY_18"], vocab["NOTE_OFF_40"], vocab["NOTE_OFF_49"], vocab["NOTE_OFF_35"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_ON_63"], vocab["VELOCITY_26"], vocab["TIME_SHIFT_40ms"], vocab["NOTE_ON_84"], vocab["VELOCITY_26"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_OFF_72"], vocab["NOTE_OFF_75"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_ON_72"], vocab["VELOCITY_21"], vocab["NOTE_ON_75"], vocab["VELOCITY_21"], vocab["TIME_SHIFT_40ms"], vocab["NOTE_OFF_84"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_ON_84"], vocab["VELOCITY_21"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_OFF_72"], vocab["NOTE_OFF_75"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_ON_72"], vocab["VELOCITY_27"], vocab["NOTE_ON_75"], vocab["VELOCITY_27"], vocab["TIME_SHIFT_40ms"], vocab["NOTE_OFF_84"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_ON_84"], vocab["VELOCITY_27"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_OFF_72"], vocab["NOTE_OFF_75"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_ON_72"], vocab["VELOCITY_22"], vocab["NOTE_ON_75"], vocab["VELOCITY_22"], vocab["TIME_SHIFT_40ms"], vocab["NOTE_OFF_84"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_ON_84"], vocab["VELOCITY_22"], vocab["TIME_SHIFT_10ms"], vocab["NOTE_OFF_72"], vocab["NOTE_OFF_75"], vocab["NOTE_OFF_60"], vocab["NOTE_OFF_63"]]

generated_sequence = generate_sequence(model, seed_prompt)
generated_tokens = [idx_to_token[token_id] for token_id in generated_sequence]

print("Generated Sequence:")
print(" ".join(generated_tokens))


tokens = generated_tokens
decode_to_midi_basic_vocab(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}cheated.mid", turn_off_notes=True, max_len=600)
decode_to_midi_basic_vocab(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}.mid", turn_off_notes=False)