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
from tokenize_midi import tokenize_basic_vocab, tokenize_basic_vocab_velocity_bins, tokenize_multi_instr_vocab
import yaml

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config("config.yaml")

###### CLEAN DATA, SPLIT INTO TEST, TRAIN, AND VAL, THEN TOKENIZE INTO IDS #########

source_dir = "datasets/raw_midi/lmd_full_direct"

tokenized_train_path = "datasets/tokenized/train"
tokenized_val_path   = "datasets/tokenized/val"
tokenized_test_path  = "datasets/tokenized/test"

# Temporary directory for tokenized files before splitting:
temp_tokenized_dir   = "datasets/tokenized/temp"

TRAIN_RATIO = config['Preprocessing']['TRAIN_RATIO']
VAL_RATIO   = config['Preprocessing']['VAL_RATIO']
TEST_RATIO  = config['Preprocessing']['TEST_RATIO']
SAMPLE_SIZE = config['Preprocessing']['SAMPLE_SIZE']
BATCH_SIZE = config['Preprocessing']['BATCH_SIZE']
VOCAB = config['VOCAB']
MAX_WORKERS = config['Preprocessing']['MAX_WORKERS']

TOTAL_FILE_COUNT = len(glob.glob(os.path.join(source_dir, "*.mid"))) #178561

for d in [
    tokenized_test_path,
    tokenized_train_path,
    tokenized_val_path,
    temp_tokenized_dir
]:
    os.makedirs(d, exist_ok=True)



tokenizer = tokenize_basic_vocab

if VOCAB == "BASIC":
    print(f"USING TOKENIZER: {VOCAB}")
    tokenizer = tokenize_basic_vocab

if VOCAB == "BASIC_VELOCITY_BINS":
    print(f"USING TOKENIZER: {VOCAB}")
    tokenizer = tokenize_basic_vocab_velocity_bins

if VOCAB == "MULTI_STR":
    print(f"USING TOKENIZER: {VOCAB}")
    tokenizer = tokenize_multi_instr_vocab


executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)

def tokenize_with_timeout(midi, timeout=10):
    """
    tokenize midi with timeout
    """
    future = executor.submit(tokenizer, midi)
    return future.result(timeout=timeout)


def process_batch(batch_size):
    """
    Process up to batch_size MIDI files and save to temp dir.
    """
    batch_files = []
    with os.scandir(source_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.lower().endswith(".mid"):
                batch_files.append(entry.path)
                if len(batch_files) >= batch_size:
                    break

    print(f"Processing {len(batch_files)} files from '{source_dir}'.")

    valid_count = 0
    corrupted_deleted = 0
    multi_track_deleted = 0
    zero_length_sequence_deleted = 0
    timeout_deleted = 0

    pbar = tqdm(total=len(batch_files), desc="Processing MIDI Files")

    for file_path in batch_files:
        try:
            midi = MidiFile(file_path)
        except Exception as e:
            print(f"Error processing '{file_path}': {e}")
            corrupted_deleted += 1
            try:
                os.remove(file_path)
            except Exception as del_err:
                print(f"Error deleting '{file_path}': {del_err}")
            pbar.update(1)
            continue

        if len(midi.tracks) == 1:
            try:
                token_sequence = tokenize_with_timeout(midi, timeout=5)
            except TimeoutError:
                print(f"Timeout tokenizing '{file_path}'")
                timeout_deleted += 1
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
                zero_length_sequence_deleted += 1
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting zero-length file '{file_path}': {e}")
        else:
            multi_track_deleted += 1
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting multi-track file '{file_path}': {e}")

        pbar.update(1)
    pbar.close()
    
    return valid_count, corrupted_deleted, multi_track_deleted, zero_length_sequence_deleted, timeout_deleted

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

    print(f"\nSplitting {valid_count} valid files...")

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


if __name__ == "__main__":
    print(f"Total number of files: {TOTAL_FILE_COUNT}")
    total_valid_count = 0
    total_corrupted_deleted = 0
    total_multi_track_deleted = 0
    total_zero_length_sequence_deleted = 0
    total_timeout_deleted = 0

    while total_valid_count < SAMPLE_SIZE:
        # process batch
        batch_size = min(BATCH_SIZE, SAMPLE_SIZE - total_valid_count)
        valid_count, corrupted_deleted, multi_track_deleted, zero_length_sequence_deleted, timeout_deleted = process_batch(batch_size)

        # update processing stats
        total_valid_count += valid_count
        total_corrupted_deleted += corrupted_deleted
        total_multi_track_deleted += multi_track_deleted
        total_zero_length_sequence_deleted += zero_length_sequence_deleted
        total_timeout_deleted += timeout_deleted

        print(f"\nProcessed {total_valid_count}/{SAMPLE_SIZE} valid files so far...\n")

        if total_valid_count + total_corrupted_deleted + total_multi_track_deleted + total_zero_length_sequence_deleted + total_timeout_deleted >= TOTAL_FILE_COUNT:
            break
    
    move_files_to_final()

    print(f"{total_valid_count} valid midi files found")
    print(f"{total_corrupted_deleted} corrupt midi files deleted")
    print(f"{total_multi_track_deleted} multi-track midi files deleted")
    print(f"{total_zero_length_sequence_deleted} zero_length midi files deleted")
    print(f"{total_timeout_deleted} midi files timed out and deleted")