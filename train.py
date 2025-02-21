import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.MusicTransformer import MusicTransformer
from data_processing.dataset import MidiDataset, collate_batch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


# dataset paths
EXPERIMENT_NAME = "40epoch_mid"

TRAIN_DATA_PATH = "datasets/tokenized/train"
VAL_DATA_PATH = "datasets/tokenized/val"
TEST_DATA_PATH = "datasets/tokenized/test"
VOCAB_PATH = "datasets/vocab/basic_vocab.json"
CHECKPOINT_DIR = f"{EXPERIMENT_NAME}/checkpoints"

os.makedirs(EXPERIMENT_NAME, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4

# Model hyperparameters
BLOCK_SIZE = 1024
N_EMBD = 1024
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.01
MAX_SEQ_LENGTH = BLOCK_SIZE

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"USING {device}") 

# Load Vocabulary
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")

# set up dataloader
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
    shuffle=False,
    collate_fn=collate_batch
)

# setup model
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



#training loop
print(f"BEGIN TRAINING MODEL WITH: {sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters")
train_losses = []
val_losses = []

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

        # gradient clipping        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
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
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch}/{NUM_EPOCHS} - Validation Loss: {avg_val_loss:.4f}")


    # Save checkpoint every epoch
    if epoch % 4 == 0 or epoch == NUM_EPOCHS:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


# Save the final model
final_model_path = os.path.join(EXPERIMENT_NAME, "model_final.pt")
torch.save(model.state_dict(), final_model_path)
print(f"Training complete. Final model saved to {final_model_path}")

# plot training/val loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Training Loss", marker="o")
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(EXPERIMENT_NAME, "loss_plot.png"))
plt.show()
