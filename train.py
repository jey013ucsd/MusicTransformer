import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.MusicTransformer import MusicTransformer
from data_processing.dataset import MidiDataset, collate_batch
from tqdm import tqdm


# -------------------------------
# Hyperparameter Setup
# -------------------------------

# dataset paths
TRAIN_DATA_PATH = "datasets/tokenized/train"
VAL_DATA_PATH = "datasets/tokenized/val"
TEST_DATA_PATH = "datasets/tokenized/test"
VOCAB_PATH = "datasets/vocab/basic_vocab.json"

# Checkpoint directory
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 2
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4

# Model hyperparameters
BLOCK_SIZE = 2048
N_EMBD = 512
N_HEAD = 8
N_LAYER = 6
DROPOUT = 0.1
MAX_SEQ_LENGTH = BLOCK_SIZE

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING {device}") 

# Load Vocabulary
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")


# -------------------------------
# Setup Dataloaders
# -------------------------------

train_dataset = MidiDataset(
    data_dir=TRAIN_DATA_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    pad_token_id=vocab["TOKEN_PAD"]
)

val_dataset = MidiDataset(
    data_dir=VAL_DATA_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    pad_token_id=vocab["TOKEN_PAD"]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch
)


# -------------------------------
# Model Setup
# -------------------------------
model = MusicTransformer(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
    max_len=MAX_SEQ_LENGTH
).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["TOKEN_PAD"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -------------------------------
# Training Loop
# -------------------------------
print(f"BEGIN TRAINING MODEL WITH: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")

for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Training Progress"):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)  # logits shape: (B, T, VOCAB_SIZE)

        # Flatten logits and targets for computing loss
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Training Loss: {avg_train_loss:.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Validation Loss: {avg_val_loss:.4f}")

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}")

# Save the final model
final_model_path = os.path.join(CHECKPOINT_DIR, "model_final.pt")
torch.save(model.state_dict(), final_model_path)
print(f"🎉 Training complete. Final model saved to {final_model_path}")
