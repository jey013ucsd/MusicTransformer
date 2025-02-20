import mido
from mido import MidiFile, tick2second
from tqdm import tqdm
import json
import os
import pickle

raw_train_path = "datasets/raw_midi/lmd_clean_1track/train"
raw_val_path = "datasets/raw_midi/lmd_clean_1track/val"
raw_test_path = "datasets/raw_midi/lmd_clean_1track/test"

tokenized_train_path = "datasets/tokenized/train"
tokenized_val_path = "datasets/tokenized/val"
tokenized_test_path = "datasets/tokenized/test"

# Load Vocab
vocab_path = "datasets/vocab/basic_vocab.json"
with open(vocab_path, "r") as f:
    vocab = json.load(f)

def tokenize_single_midi(midi_path):
    '''
    Tokenize a single midi file using defined vocabulary
    
    Returns a sequence of tokens ids
    '''
    mid = MidiFile(midi_path)
    track = mid.tracks[0]
    
    current_tempo = 500000
    token_sequence = []

    for msg in track:
        if msg.type == 'set_tempo':
            current_tempo = msg.tempo
            continue
        if msg.time > 0:
            # convert ticks to seconds
            delta_seconds = tick2second(msg.time, mid.ticks_per_beat, current_tempo)
            delta_ms = int(round(delta_seconds * 1000))
            
            # Because our TIME_SHIFT tokens only come in 10ms increments (min 10ms, max 1000ms),
            # we need to break the delta time into one or more TIME_SHIFT tokens.
            while delta_ms > 0:
                # If delta_ms is less than 10, we still output a 10ms token (our resolution is 10ms)
                if delta_ms < 10:
                    shift = 10
                else:
                    # Use multiples of 10 up to 1000ms
                    shift = min(1000, (delta_ms // 10) * 10)
                    if shift == 0:
                        shift = 10
                token = f"TIME_SHIFT_{shift}ms"
                token_sequence.append(token)
                delta_ms -= shift

        if msg.type == 'note_on':
            if msg.velocity == 0:
                token_sequence.append(f"NOTE_OFF_{msg.note}")
            else:
                token_sequence.append(f"NOTE_ON_{msg.note}")
                velocity_bin = max(1, min(32, int(round((msg.velocity / 127) * 32))))
                token_sequence.append(f"VELOCITY_{velocity_bin}")
        elif msg.type == 'note_off':
            token = f"NOTE_OFF_{msg.note}"
            token_sequence.append(token)


    token_id_sequence = []
    for t in token_sequence:
        token_id_sequence.append(vocab[t])

    return token_id_sequence

def tokenize_and_save_dataset(raw_midi_path, tokenized_output_path):
    """
    Processes all MIDI files in a directory, tokenizes them, and saves them as pickled sequences.
    
    Args:
        raw_midi_path (str): Path to the directory containing raw MIDI files.
        tokenized_output_path (str): Path where tokenized sequences will be saved.
    """
    os.makedirs(tokenized_output_path, exist_ok=True)
    
    midi_files = [f for f in os.listdir(raw_midi_path) if f.endswith(".mid") or f.endswith(".midi")]

    print(f"Processing {len(midi_files)} files from {raw_midi_path}...")

    for midi_file in tqdm(midi_files, desc=f"Tokenizing {os.path.basename(raw_midi_path)}"):
        midi_path = os.path.join(raw_midi_path, midi_file)
        tokenized_sequence = tokenize_single_midi(midi_path)

        if len(tokenized_sequence) == 0:
            print(f"Deleting empty sequence file: {midi_path}")
            os.remove(midi_path)
        else:
            # Save the tokenized sequence
            save_path = os.path.join(tokenized_output_path, f"{midi_file}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(tokenized_sequence, f)

    print(f"Tokenized dataset saved in {tokenized_output_path}")

if __name__ == "__main__":
    tokenize_and_save_dataset(raw_train_path, tokenized_train_path)
    tokenize_and_save_dataset(raw_val_path, tokenized_val_path)
    tokenize_and_save_dataset(raw_test_path, tokenized_test_path)