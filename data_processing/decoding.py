import os
import json
import torch
import torch.nn.functional as F
from models.MusicTransformer import MusicTransformer
import mido
from mido import MidiFile, MidiTrack, Message, second2tick, MetaMessage

BASIC_VOCAB_PATH  = "models/vocab/basic_vocab.json"

with open(BASIC_VOCAB_PATH, "r") as f:
    basic_vocab = json.load(f)
basic_id_to_token = {v: k for k, v in basic_vocab.items()}


def decode_to_tokens_basic_vocab(token_sequence):
    return [basic_id_to_token[token_id] for token_id in token_sequence]


def off_notes(mid, max_length_ms=350, tempo=500000):
    """
    Forces notes to max_length_ms length
    """
    if not mid.tracks:
        return
    
    track = mid.tracks[0]
    ticks_per_beat = mid.ticks_per_beat
    
    # Convert from delta to absolute times
    abs_events = []
    current_abs_ticks = 0
    for msg in track:
        current_abs_ticks += msg.time
        abs_events.append((current_abs_ticks, msg))
    
    note_on_dict = {}
    
    # Convert max_length_ms to ticks
    max_length_ticks = int(round(second2tick(max_length_ms / 1000.0, ticks_per_beat, tempo)))
    

    processed_events = []
    
    for (abs_t, msg) in abs_events:
        # if any notes currently on exceed max_length, insert note_off
        notes_to_off = []
        for note, (on_ticks, channel, velocity) in note_on_dict.items():
            if abs_t >= on_ticks + max_length_ticks:
                # It's past the limit; force note_off at (on_ticks + max_length_ticks)
                forced_off_time = on_ticks + max_length_ticks
                # Insert a note_off event there
                forced_off_msg = Message('note_off', note=note, velocity=0, time=0, channel=channel)
                notes_to_off.append((forced_off_time, forced_off_msg))
        
        # Insert each forced note_off, sorted by forced_off_time
        # (In practice, they may all be the same time if they exceed together)
        for (off_t, off_msg) in sorted(notes_to_off, key=lambda x: x[0]):
            if off_msg.note in note_on_dict:
                del note_on_dict[off_msg.note]  # remove from dict
            processed_events.append((off_t, off_msg))
        
        # Now handle the current event
        # If it's note_on, record it
        # If it's note_off, remove from dict
        if msg.type == 'note_on' and msg.velocity > 0:
            note_on_dict[msg.note] = (abs_t, msg.channel, msg.velocity)
            processed_events.append((abs_t, msg))
        elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
            # It's a real note_off
            note = msg.note
            if note in note_on_dict:
                del note_on_dict[note]
            processed_events.append((abs_t, msg))
        else:
            processed_events.append((abs_t, msg))

    # force close notes at end
    final_time = processed_events[-1][0] if processed_events else 0
    for note, (on_ticks, channel, velocity) in note_on_dict.items():

        forced_off_time = min(final_time, on_ticks + max_length_ticks)
        forced_off_msg = Message('note_off', note=note, velocity=0, channel=channel)
        processed_events.append((forced_off_time, forced_off_msg))
    
    
    # Convert back to delta times
    processed_events.sort(key=lambda x: x[0])
    new_track = []
    last_t = 0
    for (abs_t, msg) in processed_events:
        delta = abs_t - last_t
        new_msg = msg.copy(time=delta)
        new_track.append(new_msg)
        last_t = abs_t
    
    track.clear()
    track.extend(new_track)

def decode_to_midi_basic_vocab(token_sequence, save_path, ticks_per_beat=480, tempo=500000, turn_off_notes=False, max_len=450):
    """
    Decodes tokens with basic vocab and converst to midi file

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

    if turn_off_notes:
        off_notes(mid, max_length_ms=max_len)
        
    mid.save(save_path)
    print(f"MIDI file saved to {save_path}")


def decode_to_midi_basic_vocab_velocity_bins(token_sequence, save_path, ticks_per_beat=480, tempo=500000, turn_off_notes=False, max_len=450):
    """
    Decodes tokens with basic_vocab_veocity_bins and converts to midi file
    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    
    accumulated_time_ticks = 0
    current_velocity = 64  # default velocity
    
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
            ticks = second2tick(seconds, ticks_per_beat, tempo)
            accumulated_time_ticks += int(round(ticks))
            i += 1

        elif token.startswith("VELOCITY_"):
            try:
                velocity_bin = int(token[len("VELOCITY_"):])
            except ValueError:
                print(f"Warning: Could not parse velocity token: {token}. Using default velocity 64.")
                current_velocity = 64
            else:
                current_velocity = int(round((velocity_bin / 32) * 127))
            i += 1

        elif token.startswith("NOTE_ON_"):
            try:
                note = int(token[len("NOTE_ON_"):])
            except ValueError:
                print(f"Warning: Could not parse NOTE_ON token: {token}. Skipping.")
                i += 1
                continue
            msg = Message("note_on", note=note, velocity=current_velocity, time=accumulated_time_ticks)
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

        else:
            print(f"Warning: Unrecognized token: {token}. Skipping.")
            i += 1

    if turn_off_notes:
        off_notes(mid, max_length_ms=max_len)
        
    mid.save(save_path)
    print(f"MIDI file saved to {save_path}")
