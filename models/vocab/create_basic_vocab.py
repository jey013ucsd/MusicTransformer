import json

def build_vocab():
    """
    Build a basic vocabulary for the Music Transformer:
      - 128 NOTE_ON events,
      - 128 NOTE_OFF events,
      - 100 TIME_SHIFT events (10ms increments up to 1000ms),
      - 32 VELOCITY events,
      - Special tokens: TOKEN_PAD
      
    save to vocab.json
    """
    vocab = {}
    idx = 0

    # NOTE_ON tokens: for each pitch 0-127
    for pitch in range(128):
        token = f"NOTE_ON_{pitch}"
        vocab[token] = idx
        idx += 1

    # NOTE_OFF tokens: for each pitch 0-127
    for pitch in range(128):
        token = f"NOTE_OFF_{pitch}"
        vocab[token] = idx
        idx += 1

    # TIME_SHIFT tokens: 100 tokens, from 10ms to 1000ms
    for t in range(1, 101):
        token = f"TIME_SHIFT_{t * 10}ms"
        vocab[token] = idx
        idx += 1

    # VELOCITY tokens: 32 tokens representing expressive dynamics
    for level in range(1, 33):
        token = f"VELOCITY_{level}"
        vocab[token] = idx
        idx += 1

    # Padding and end-of-sequence tokens
    vocab["TOKEN_PAD"] = idx
    idx += 1

    return vocab

if __name__ == "__main__":
    vocab = build_vocab()
    print(f"Vocabulary size: {len(vocab)}")
    
    # Save vocab as json file
    with open("models/vocab/basic_vocab.json", "w") as f:
        json.dump(vocab, f, indent=4)
