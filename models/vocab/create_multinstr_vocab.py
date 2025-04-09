import json

def build_vocab():
    """
    Build a vocabulary for the Music Transformer with multiinstrument support and specific drum tokens:
      - 128 NOTE_ON events
      - 128 NOTE_OFF events
      - 100 TIME_SHIFT events (10ms increments up to 1000ms)
      - 16 VELOCITY events
      - 128 INSTRUMENT change events
      - 128 DRUM events
      - Special tokens: TOKEN_PAD
      
    save to models/vocab/multi_instr_vocab.json
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
    for level in range(1, 17):
        token = f"VELOCITY_{level}"
        vocab[token] = idx
        idx += 1

    # INSTRUMENT tokens (0-127 MIDI programs)
    for instr in range(128):
        token = f"INSTRUMENT_{instr}"
        vocab[token] = idx
        idx += 1

    # DRUM_ON tokens (0-127 drum sets)
    for note in range(128):  # or maybe a subset for common drum notes
        token = f"DRUM_ON_{note}"
        vocab[token] = idx
        idx += 1

    # DRUM_OFF tokens (0-127 drum sets)
    for note in range(128):  # or maybe a subset for common drum notes
        token = f"DRUM_OFF_{note}"
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
    with open("models/vocab/multi_instr_vocab.json", "w") as f:
        json.dump(vocab, f, indent=4)
