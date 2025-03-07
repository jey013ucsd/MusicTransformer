import os
import random
import shutil
import glob
import mido
from mido import MidiFile, tick2second
from tqdm import tqdm
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, TimeoutError

###### CLEAN DATA, SPLIT INTO TEST, TRAIN, AND VAL, THEN TOKENIZE INTO IDS #########

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
SAMPLE_SIZE = 178561
VOCAB_PATH  = "datasets/vocab/basic_vocab.json"

timeout_count = 0
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

executor = ProcessPoolExecutor(max_workers=1)

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

            while delta_ms > 0:
                if delta_ms < 10:
                    shift = 10
                else:
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
            token_sequence.append(f"NOTE_OFF_{msg.note}")

    token_id_sequence = [vocab[t] for t in token_sequence]

    # Trim leading/trailing time shifts
    start = 0
    while start < len(token_id_sequence) and id_to_token[token_id_sequence[start]].startswith("TIME_SHIFT_"):
        start += 1
    end = len(token_id_sequence)
    while end > start and id_to_token[token_id_sequence[end-1]].startswith("TIME_SHIFT_"):
        end -= 1
    token_id_sequence = token_id_sequence[start:end]
    return token_id_sequence


def tokenize_with_timeout(midi, timeout=10):
    """
    tokenize midi with timeout
    """
    future = executor.submit(tokenize_single_midi, midi)
    return future.result(timeout=timeout)


def get_batch_files(source_dir, batch_size):
    """
    Retrieve up to batch_size file paths from source_dir
    """
    batch_files = []
    with os.scandir(source_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".mid"):
                batch_files.append(entry.path)
                if len(batch_files) >= batch_size:
                    break
    random.shuffle(batch_files)
    return batch_files


def process_dataset(batch_size):
    """
    Process up to batch_size MIDI files and save to temp dir.
    """
    batch_files = get_batch_files(source_dir, batch_size)
    print(f"Processing {len(batch_files)} files from '{source_dir}'.")

    valid_count = 0
    corrupted_deleted = 0
    multi_track_deleted = 0
    zero_length_sequence_deleted = 0

    pbar = tqdm(total=len(batch_files), desc="Processing MIDI Files")

    for file_path in batch_files:
        try:
            midi = MidiFile(file_path)
        except Exception as e:
            print(f"Error processing '{file_path}': {e}")
            try:
                os.remove(file_path)
                corrupted_deleted += 1
            except Exception as del_err:
                print(f"Error deleting '{file_path}': {del_err}")
            pbar.update(1)
            continue

        if len(midi.tracks) == 1:
            try:
                token_sequence = tokenize_with_timeout(midi, timeout=5)
            except TimeoutError:
                print(f"Timeout tokenizing '{file_path}'. Skipping.")
                timeout_count += 1
                try:
                    os.remove(file_path)
                except Exception as rm_err:
                    print(f"Error removing '{file_path}': {rm_err}")
                pbar.update(1)
                continue
            except Exception as e:
                print(f"Error tokenizing '{file_path}': {e}")
                pbar.update(1)
                continue

            if len(token_sequence) > 10:
                temp_pkl_path = os.path.join(
                    temp_tokenized_dir,
                    os.path.basename(file_path) + ".pkl"
                )
                with open(temp_pkl_path, "wb") as f:
                    pickle.dump(token_sequence, f)
                try:
                    os.remove(file_path)
                except Exception as rm_err:
                    print(f"Error removing '{file_path}': {rm_err}")
                valid_count += 1
            else:
                try:
                    os.remove(file_path)
                    zero_length_sequence_deleted += 1
                except Exception as e:
                    print(f"Error deleting zero-length file '{file_path}': {e}")
        else:
            try:
                os.remove(file_path)
                multi_track_deleted += 1
            except Exception as e:
                print(f"Error deleting multi-track file '{file_path}': {e}")

        pbar.update(1)
    pbar.close()
    return valid_count

def move_files_to_final():
    """
    Move processed .pkl files from temp to final train/val/test splits.
    """
    all_pkl_files = glob.glob(os.path.join(temp_tokenized_dir, "*.pkl"))
    random.shuffle(all_pkl_files)

    valid_count = len(all_pkl_files)

    train_count = int(valid_count * TRAIN_RATIO)
    val_count   = int(valid_count * VAL_RATIO)
    test_count  = valid_count - train_count - val_count

    print(f"\nSplitting {valid_count} valid files into:")
    print(f"  Train: {train_count}")
    print(f"  Val:   {val_count}")
    print(f"  Test:  {test_count}")

    train_split = all_pkl_files[:train_count]
    val_split   = all_pkl_files[train_count : train_count + val_count]
    test_split  = all_pkl_files[train_count + val_count : ]

    def handle_split(split_list, tokenized_dir):
        for pkl_path in split_list:
            filename = os.path.basename(pkl_path)
            shutil.move(pkl_path, os.path.join(tokenized_dir, filename))

    handle_split(train_split, tokenized_train_path)
    handle_split(val_split, tokenized_val_path)
    handle_split(test_split, tokenized_test_path)

    print("\n--- SPLIT COMPLETED ---")
    print(f"Total processed files: {valid_count}")

if __name__ == "__main__":
    total_processed = 0
    TOTAL_SAMPLE_SIZE = 178561
    BATCH_SIZE = 5000

    while total_processed < TOTAL_SAMPLE_SIZE:
        batch_size = min(BATCH_SIZE, TOTAL_SAMPLE_SIZE - total_processed)
        processed = process_dataset(batch_size)
        total_processed += processed

        print(f"\nProcessed {total_processed}/{TOTAL_SAMPLE_SIZE} files so far...\n")
        move_files_to_final()

    print(f"{timeout_count} midi files timed out and deleted")