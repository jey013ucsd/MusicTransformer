import os
import json
import torch
import torch.nn.functional as F

# Import your model class (adjust the import as needed)
from models.MusicTransformer import MusicTransformer  # Update with the actual module name

# -------------------------------
# Hyperparameters & File Paths
# -------------------------------

# Paths
VOCAB_PATH = "datasets/vocab/basic_vocab.json"
CHECKPOINT_PATH = "checkpoints/model_final.pt"  # Change if you want a specific checkpoint

# Model hyperparameters (must match those used during training)
VOCAB_SIZE = None  # will be determined from vocab
N_EMBD = 128       # For testing/inference, use the same as training
N_HEAD = 4
N_LAYER = 1
DROPOUT = 0.001
MAX_SEQ_LENGTH = 128  # Should be same as block_size used during training

# Inference hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_GENERATED_TOKENS = 128  # maximum number of tokens to generate
TEMPERATURE = 1.0           # sampling temperature
TOP_K = 0                   # if >0, perform top-k sampling

# -------------------------------
# Load Vocabulary & Create Reverse Mapping
# -------------------------------
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

# Create reverse mapping: token id -> token string
idx_to_token = {int(v): k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")

# -------------------------------
# Instantiate the Model and Load Checkpoint
# -------------------------------
model = MusicTransformer(
    vocab_size=VOCAB_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
    max_len=MAX_SEQ_LENGTH
).to(DEVICE)

# Load the pretrained weights
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print(f"Loaded model from {CHECKPOINT_PATH}")

# -------------------------------
# Inference Function
# -------------------------------
def generate_sequence(model, seed_tokens, max_tokens=MAX_GENERATED_TOKENS, temperature=TEMPERATURE, top_k=TOP_K):
    """
    Autoregressively generate a sequence of token ids starting from seed_tokens.
    
    Args:
        model: The MusicTransformer model.
        seed_tokens (list[int]): The starting sequence of token ids.
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_k (int): If greater than 0, only sample from the top k tokens.
        
    Returns:
        List[int]: The generated sequence of token ids.
    """
    model.eval()
    generated = seed_tokens.copy()
    for _ in range(max_tokens):
        # Use only the most recent max sequence length tokens
        input_seq = torch.tensor(generated[-MAX_SEQ_LENGTH:], dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(input_seq)  # shape: (1, T, VOCAB_SIZE)
        # Get the logits for the last token in the sequence
        logits = logits[0, -1, :] / temperature
        # Optionally perform top-k filtering
        if top_k > 0:
            topk_logits, topk_indices = torch.topk(logits, top_k)
            probs = torch.zeros_like(logits)
            probs[topk_indices] = F.softmax(topk_logits, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
        # Sample the next token
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        
        # Stop if TOKEN_END is generated
        if idx_to_token.get(next_token, "") == "TOKEN_END":
            break
    return generated

# -------------------------------
# Define a Seed Prompt
# -------------------------------
# For a simple test, you can define a seed prompt as a list of token ids.
# Here we assume that the seed prompt is just one token (e.g. a NOTE_ON event).
# You might want to change this to a more meaningful prompt.
seed_prompt = [vocab["NOTE_ON_60"]]  # e.g., start with NOTE_ON for Middle C

# -------------------------------
# Generate a Sequence
# -------------------------------
generated_sequence = generate_sequence(model, seed_prompt)
generated_tokens = [idx_to_token[token_id] for token_id in generated_sequence]

print("Generated Sequence:")
print(" ".join(generated_tokens))
