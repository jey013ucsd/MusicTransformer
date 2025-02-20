import os
import random
import shutil
import glob
import mido
from tqdm import tqdm
import argparse

source_dir = "datasets/raw_midi/lmd_full_direct"
dest_dir = "datasets/raw_midi/lmd_clean_1track"

train_dir = "datasets/raw_midi/lmd_clean_1track/train"
val_dir = "datasets/raw_midi/lmd_clean_1track/val"
test_dir = "datasets/raw_midi/lmd_clean_1track/test"

# Train-Test Split Ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

def get_valid_midi_files(sample_size=10000):
    '''
    We only want to train on single track midi files.
    '''
    os.makedirs(dest_dir, exist_ok=True)
    
    # Get all MIDI files in the source directory.
    all_files = glob.glob(os.path.join(source_dir, "*.mid"))
    total_files = len(all_files)
    print(f"Found {total_files} MIDI files in '{source_dir}'.")
    
    valid_count = 0
    random.shuffle(all_files)
    pbar = tqdm(total=sample_size, desc="Valid MIDI Files Processed")
    valid_deleted = 0
    corrupted_deleted = 0
    multi_track_deleted = 0
    for file_path in all_files:
        if valid_count >= sample_size:
            break

        try:
            midi = mido.MidiFile(file_path)
        except Exception as e:
            print(f"Error processing '{file_path}': {e}")
            # Delete the file from the source directory if it cannot be processed.
            try:
                corrupted_deleted += 1
                os.remove(file_path)
            except Exception as del_err:
                print(f"Error deleting '{file_path}': {del_err}")
            continue
        
        # Check if the file has exactly one track.
        if len(midi.tracks) == 1:
            filename = os.path.basename(file_path)
            dest_file = os.path.join(dest_dir, filename)
            try:
                shutil.copy(file_path, dest_file)
                valid_count += 1
                pbar.update(1)
            except Exception as e:
                print(f"Error copying '{file_path}' to '{dest_file}': {e}")
                continue

            # Delete the file from the source directory after successful copy.
            try:
                os.remove(file_path)
                valid_deleted += 1
            except Exception as e:
                print(f"Error deleting '{file_path}': {e}")
        else:
            # Delete files with more than one track.
            try:
                os.remove(file_path)
                multi_track_deleted += 1
            except Exception as e:
                print(f"Error deleting multi-track file '{file_path}': {e}")
    
    pbar.close()
    
    if valid_count < sample_size:
        print(f"Only found {valid_count} valid MIDI files out of {total_files} files.")
    else:
        print(f"Successfully processed {valid_count} valid single-track MIDI files to '{dest_dir}'.")
        print(f"Successfully deleted {multi_track_deleted} multi-track MIDI files from '{source_dir}'.")
        print(f"Successfully deleted {valid_deleted} valid MIDI files from '{source_dir}'.")
        print(f"Successfully deleted {corrupted_deleted} corrupt MIDI files from '{source_dir}'.")

def create_splits():
    '''
    Create Test, Train, and Validation splits
    '''
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)

    all_files = [f for f in os.listdir(dest_dir) if f.endswith(".mid") or f.endswith(".midi")]
    random.shuffle(all_files)
    
    total_files = len(all_files)
    train_count = int(total_files * TRAIN_RATIO)
    val_count = int(total_files * VAL_RATIO)

    print(f"Creating Test, Train, and Validation splits from {total_files} found")
    
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]
    
    def move_files(file_list, destination_folder):
        for file_name in file_list:
            src_path = os.path.join(dest_dir, file_name)
            dst_path = os.path.join(destination_folder, file_name)
            shutil.move(src_path, dst_path)
    
    # Move files into respective directories
    move_files(train_files, train_dir)
    move_files(val_files, val_dir)
    move_files(test_files, test_dir)
    
    # Print summary
    print(f"✅ Dataset split complete!")
    print(f"📂 Total files: {total_files}")
    print(f"🔹 Training set: {len(train_files)} files")
    print(f"🔹 Validation set: {len(val_files)} files")
    print(f"🔹 Test set: {len(test_files)} files")


if __name__ == "__main__":
    get_valid_midi_files()
    create_splits()