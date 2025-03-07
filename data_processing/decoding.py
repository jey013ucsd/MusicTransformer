import os
import json
import torch
import torch.nn.functional as F
from models.MusicTransformer import MusicTransformer
import mido
from mido import MidiFile, MidiTrack, Message, second2tick

BASIC_VOCAB_PATH  = "datasets/vocab/basic_vocab.json"

with open(BASIC_VOCAB_PATH, "r") as f:
    basic_vocab = json.load(f)
basic_id_to_token = {v: k for k, v in basic_vocab.items()}


def decode_to_tokens_basic_vocab(token_sequence):
    return [basic_id_to_token[token_id] for token_id in token_sequence]


def decode_to_midi_basic_vocab(token_sequence, save_path, ticks_per_beat=480, tempo=500000, turn_off_notes=False):
    """
    Decodes tokens and converst to midi file

    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    

    accumulated_time_ticks = 0

    i = 0
    while i < len(token_sequence):
        token = token_sequence[i]

        if token.startswith("TIME_SHIFT_"):
            try:
                ms_value = int(token[len("TIME_SHIFT_"):-2])
            except ValueError:
                print(f"Warning: Could not parse time shift token: {token}. Skipping.")
                i += 1
                continue

            seconds = ms_value / 1000.0
            # Convert seconds to ticks
            ticks = second2tick(seconds, ticks_per_beat, tempo)
            accumulated_time_ticks += int(round(ticks))
            i += 1

        elif token.startswith("NOTE_ON_"):
            try:
                note = int(token[len("NOTE_ON_"):])
            except ValueError:
                print(f"Warning: Could not parse NOTE_ON token: {token}. Skipping.")
                i += 1
                continue

            # every note should correspond with a velocity
            if i + 1 < len(token_sequence) and token_sequence[i+1].startswith("VELOCITY_"):
                velocity_token = token_sequence[i+1]
                try:
                    velocity_bin = int(velocity_token[len("VELOCITY_"):])
                except ValueError:
                    print(f"Warning: Could not parse velocity token: {velocity_token}. Using default velocity 64.")
                    velocity = 64
                else:
                    velocity = int(round((velocity_bin / 32) * 127))
                msg = Message("note_on", note=note, velocity=velocity, time=accumulated_time_ticks)
                track.append(msg)
                accumulated_time_ticks = 0
                i += 2
            else:
                print(f"Warning: Missing VELOCITY token after {token}. Using default velocity 64.")
                msg = Message("note_on", note=note, velocity=64, time=accumulated_time_ticks)
                track.append(msg)
                accumulated_time_ticks = 0
                i += 1

        elif token.startswith("NOTE_OFF_"):
            try:
                note = int(token[len("NOTE_OFF_"):])
            except ValueError:
                print(f"Warning: Could not parse NOTE_OFF token: {token}. Skipping.")
                i += 1
                continue
            msg = Message("note_off", note=note, velocity=0, time=accumulated_time_ticks)
            track.append(msg)
            accumulated_time_ticks = 0
            i += 1

        elif token.startswith("VELOCITY_"):
            # skip unexpected velocity tokens
            print(f"Warning: Unexpected token {token} encountered; skipping.")
            i += 1

        else:
            print(f"Warning: Unrecognized token: {token}. Skipping.")
            i += 1

    mid.save(save_path)
    print(f"MIDI file saved to {save_path}")