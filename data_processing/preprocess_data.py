import os
import random
import shutil
import glob
import mido
from mido import MidiFile, tick2second
from tqdm import tqdm
import json
import pickle

###### CLEAN DATA, SPLIT INTO TEST, TRAIN, AND VAL, THEN TOKENIZE INTO IDS #########

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SAMPLE_SIZE = 178561
VOCAB_PATH = "datasets/vocab/basic_vocab.json"

# Directories
source_dir = "datasets/raw_midi/lmd_full_direct"

# Final tokenized data directories:
tokenized_train_path = "datasets/tokenized/train"
tokenized_val_path   = "datasets/tokenized/val"
tokenized_test_path  = "datasets/tokenized/test"

# Temporary directory for tokenized files before splitting:
temp_tokenized_dir   = "datasets/tokenized/temp"

for d in [
    tokenized_test_path, 
    tokenized_train_path, 
    tokenized_val_path, 
    temp_tokenized_dir
]:
    os.makedirs(d, exist_ok=True)

# Load vocab
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

id_to_token = {v: k for k, v in vocab.items()}

def tokenize_single_midi(mid):
    """
    Tokenize into ids a single MIDI file using defined vocabulary.
    """
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

    token_id_sequence = [vocab[t] for t in token_sequence]

    # Trim leading/trailing time shifts
    while token_id_sequence and id_to_token[token_id_sequence[0]].startswith("TIME_SHIFT_"):
        token_id_sequence.pop(0)
    while token_id_sequence and id_to_token[token_id_sequence[-1]].startswith("TIME_SHIFT_"):
        token_id_sequence.pop(-1)
        
    return token_id_sequence

def process_dataset():
    all_files = glob.glob(os.path.join(source_dir, "*.mid"))
    random.shuffle(all_files)

    print(f"Found {len(all_files)} MIDI files in '{source_dir}'.")
    
    valid_pkl_files = []
    corrupted_deleted = 0
    multi_track_deleted = 0
    zero_length_sequence_deleted = 0

    pbar = tqdm(total=SAMPLE_SIZE, desc="Valid MIDI Files Processed")

    for file_path in all_files:
        if len(valid_pkl_files) >= SAMPLE_SIZE:
            break

        # Try to parse the MIDI
        try:
            midi = MidiFile(file_path)
        except Exception as e:
            # Delete corrupted file
            print(f"Error processing '{file_path}': {e}")
            try:
                os.remove(file_path)
                corrupted_deleted += 1
            except Exception as del_err:
                print(f"Error deleting '{file_path}': {del_err}")
            continue
        
        if len(midi.tracks) == 1:
            token_sequence = tokenize_single_midi(midi)
            
            if len(token_sequence) > 10:
                # Save tokenized .pkl to TEMP directory immediately
                temp_pkl_path = os.path.join(temp_tokenized_dir, os.path.basename(file_path) + ".pkl")
                with open(temp_pkl_path, "wb") as f:
                    pickle.dump(token_sequence, f)

                # Remove the original MIDI
                os.remove(file_path)

                valid_pkl_files.append(temp_pkl_path)
                pbar.update(1)
            else:
                # remove too short files
                try:
                    os.remove(file_path)
                    zero_length_sequence_deleted += 1
                except Exception as e:
                    print(f"Error deleting zero-length file '{file_path}': {e}")
        else:
            # remove multi-track file
            try:
                os.remove(file_path)
                multi_track_deleted += 1
            except Exception as e:
                print(f"Error deleting multi-track file '{file_path}': {e}")

    pbar.close()
    
    # Final count of valid files
    valid_count = len(valid_pkl_files)
    if valid_count < SAMPLE_SIZE:
        print(f"Only {valid_count} files found")
    
    # Split after we know how many valid tokenized files we have
    train_count = int(valid_count * TRAIN_RATIO)
    val_count   = int(valid_count * VAL_RATIO)
    test_count  = valid_count - train_count - val_count

    print(f"\nSplitting {valid_count} valid files into:")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")

    train_split = valid_pkl_files[:train_count]
    val_split   = valid_pkl_files[train_count : train_count + val_count]
    test_split  = valid_pkl_files[train_count + val_count : ]

    def handle_split(split_list, tokenized_dir):
        for pkl_path in split_list:
            filename = os.path.basename(pkl_path)
            shutil.move(pkl_path, os.path.join(tokenized_dir, filename))

    # Move each split from temp to final
    handle_split(train_split, tokenized_train_path)
    handle_split(val_split, tokenized_val_path)
    handle_split(test_split, tokenized_test_path)

    print("\n--- SUMMARY ---")
    print(f"Total valid files used: {valid_count}")
    print(f"Corrupted deleted:      {corrupted_deleted}")
    print(f"Multi-track deleted:    {multi_track_deleted}")
    print(f"Zero-length deleted:    {zero_length_sequence_deleted}")

if __name__ == "__main__":
    process_dataset()
