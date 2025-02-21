import os
import json
import torch
import torch.nn.functional as F
from models.MusicTransformer import MusicTransformer
import mido
from mido import MidiFile, MidiTrack, Message, second2tick

# Paths
EXPERIMENT_NAME = "40epoch_mid"
OUTPUT_NAME = "decoded_output"
VOCAB_PATH = "datasets/vocab/basic_vocab.json"
CHECKPOINT_PATH = f"{EXPERIMENT_NAME}/checkpoints/model_epoch_32.pt"

N_EMBD = 1024
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.01
MAX_SEQ_LENGTH = 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GENERATED_TOKENS = 512  # maximum number of tokens to generate
TEMPERATURE = 1.0
TOP_K = 0



with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

idx_to_token = {int(v): k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")



model = MusicTransformer(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
    max_len=MAX_SEQ_LENGTH
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
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


def decode_token_sequence(token_sequence, output_midi_path, ticks_per_beat=480, tempo=500000):
    """
    Decodes tokens and converst to midi file

    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    

    accumulated_time_ticks = 0
    i = 0
    while i < len(token_sequence):
        token = token_sequence[i]

        if token.startswith("TIME_SHIFT_"):
            try:
                ms_value = int(token[len("TIME_SHIFT_"):-2])
            except ValueError:
                print(f"Warning: Could not parse time shift token: {token}. Skipping.")
                i += 1
                continue
            seconds = ms_value / 1000.0
            # Convert seconds to ticks
            ticks = second2tick(seconds, ticks_per_beat, tempo)
            accumulated_time_ticks += int(round(ticks))
            i += 1

        elif token.startswith("NOTE_ON_"):
            try:
                note = int(token[len("NOTE_ON_"):])
            except ValueError:
                print(f"Warning: Could not parse NOTE_ON token: {token}. Skipping.")
                i += 1
                continue

            # every note should correspond with a velocity
            if i + 1 < len(token_sequence) and token_sequence[i+1].startswith("VELOCITY_"):
                velocity_token = token_sequence[i+1]
                try:
                    velocity_bin = int(velocity_token[len("VELOCITY_"):])
                except ValueError:
                    print(f"Warning: Could not parse velocity token: {velocity_token}. Using default velocity 64.")
                    velocity = 64
                else:
                    velocity = int(round((velocity_bin / 32) * 127))
                msg = Message("note_on", note=note, velocity=velocity, time=accumulated_time_ticks)
                track.append(msg)
                accumulated_time_ticks = 0
                i += 2
            else:
                print(f"Warning: Missing VELOCITY token after {token}. Using default velocity 64.")
                msg = Message("note_on", note=note, velocity=64, time=accumulated_time_ticks)
                track.append(msg)
                accumulated_time_ticks = 0
                i += 1

        elif token.startswith("NOTE_OFF_"):
            try:
                note = int(token[len("NOTE_OFF_"):])
            except ValueError:
                print(f"Warning: Could not parse NOTE_OFF token: {token}. Skipping.")
                i += 1
                continue
            msg = Message("note_off", note=note, velocity=0, time=accumulated_time_ticks)
            track.append(msg)
            accumulated_time_ticks = 0
            i += 1

        elif token.startswith("VELOCITY_"):
            # skip unexpected velocity tokens
            print(f"Warning: Unexpected token {token} encountered; skipping.")
            i += 1

        else:
            print(f"Warning: Unrecognized token: {token}. Skipping.")
            i += 1

    mid.save(output_midi_path)
    print(f"MIDI file saved as {output_midi_path}")


#input sequence
seed_prompt = [
    # Ascending Scale
    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],  # C4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_60"],

    vocab["NOTE_ON_62"], vocab["VELOCITY_20"],  # D4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_62"],

    vocab["NOTE_ON_64"], vocab["VELOCITY_20"],  # E4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_64"],

    vocab["NOTE_ON_65"], vocab["VELOCITY_20"],  # F4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_65"],

    vocab["NOTE_ON_67"], vocab["VELOCITY_20"],  # G4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_67"],

    vocab["NOTE_ON_69"], vocab["VELOCITY_20"],  # A4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_69"],

    vocab["NOTE_ON_71"], vocab["VELOCITY_20"],  # B4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_71"],

    vocab["NOTE_ON_72"], vocab["VELOCITY_20"],  # C5 (top)
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_72"],

    # Descending Scale
    vocab["NOTE_ON_71"], vocab["VELOCITY_20"],  # B4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_71"],

    vocab["NOTE_ON_69"], vocab["VELOCITY_20"],  # A4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_69"],

    vocab["NOTE_ON_67"], vocab["VELOCITY_20"],  # G4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_67"],

    vocab["NOTE_ON_65"], vocab["VELOCITY_20"],  # F4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_65"],

    vocab["NOTE_ON_64"], vocab["VELOCITY_20"],  # E4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_64"],

    vocab["NOTE_ON_62"], vocab["VELOCITY_20"],  # D4
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_62"],

    vocab["NOTE_ON_60"], vocab["VELOCITY_20"],  # C4 (back to root)
    vocab["TIME_SHIFT_400ms"],
    vocab["NOTE_OFF_60"],
]




generated_sequence = generate_sequence(model, seed_prompt)
generated_tokens = [idx_to_token[token_id] for token_id in generated_sequence]

print("Generated Sequence:")
print(" ".join(generated_tokens))


tokens = generated_tokens
decode_token_sequence(tokens, f"{EXPERIMENT_NAME}/{OUTPUT_NAME}.mid")