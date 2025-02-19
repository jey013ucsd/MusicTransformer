import urllib.request
import tarfile
import os
import mido
import glob
import pickle
import numpy as np
import random
import shutil
from skimage.feature import hog
from functools import partial
import glob
import mido
import tqdm
from tqdm import tqdm

dataset_urls = {
    "LakhMIDI": "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz",
}

def get_dataset(dataset_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "datasets", 'raw_midi')
    os.makedirs(dataset_path, exist_ok=True)

    tar_path = os.path.join(dataset_path, dataset_name + ".tar.gz")

    print("Downloading", dataset_name)
    urllib.request.urlretrieve(dataset_urls[dataset_name], tar_path)

    print("Extracting", dataset_name)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dataset_path)

    os.remove(tar_path)
    print(f"{dataset_name} downloaded and extracted successfully.")
    
def create_direct_dataset():
    source_dir = 'datasets/raw_midi/lmd_full'
    dest_dir = 'datasets/raw_midi/lmd_full_direct'
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Find all .mid files in all in lmd_full
    pattern = os.path.join(source_dir, '**', '*.mid')
    mid_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(mid_files)} MIDI files to copy.")

    # Copy all files to lmd_full_direct
    for file_path in tqdm(mid_files, desc="Copying MIDI files"):
        filename = os.path.basename(file_path)
        dest_file = os.path.join(dest_dir, filename)
        shutil.copy(file_path, dest_file)
    
    print("All files have been copied.")

if __name__ == "__main__":
    get_dataset("LakhMIDI")
    create_direct_dataset()
    print("Done")
