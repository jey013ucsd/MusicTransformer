import os
import pickle
import random
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MidiDataset(Dataset):
    """
    dataset class for midi files

    """
    def __init__(self, data_dir, max_seq_length, pad_token_id):
        self.data_dir = data_dir
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        
        self.file_paths = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.pkl')
        ]
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        
        # Load the tokenized sequence
        file_path = self.file_paths[idx]
        with open(file_path, "rb") as f:
            token_sequence = pickle.load(f)
        
        sequence_length = len(token_sequence)
        desired_length = self.max_seq_length + 1
        

        if sequence_length < desired_length:
            token_sequence = token_sequence + [self.pad_token_id] * (desired_length - sequence_length)
        else:
            start_idx = random.randint(0, max(1, sequence_length - desired_length))
            token_sequence = token_sequence[start_idx: start_idx + desired_length]
        
        # Create input and target pairs
        x = torch.tensor(token_sequence[:-1], dtype=torch.long)
        y = torch.tensor(token_sequence[1:], dtype=torch.long)
        return x, y


def collate_batch(batch):
    """
    Collate function to batch variable-length sequences.
    
    """
    xs, ys = zip(*batch)
    padded_xs = pad_sequence(xs, batch_first=True, padding_value=0)
    padded_ys = pad_sequence(ys, batch_first=True, padding_value=0)
    return padded_xs, padded_ys