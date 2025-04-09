import mido
from mido import MidiFile, tick2second
from tqdm import tqdm
import json
import os
import pickle
from collections import defaultdict

BASIC_VOCAB_PATH  = "models/vocab/basic_vocab.json"
MULTI_INSTR_VOCAB_PATH = "models/vocab/multi_instr_vocab.json"

with open(BASIC_VOCAB_PATH, "r") as f:
    basic_vocab = json.load(f)
basic_id_to_token = {v: k for k, v in basic_vocab.items()}

with open(MULTI_INSTR_VOCAB_PATH, "r") as f:
    multi_instr_vocab = json.load(f)
multi_instr_id_to_token = {v: k for k, v in multi_instr_vocab.items()}


def tokenize_basic_vocab(mid):
    """
    Tokenize into ids a single MIDI file using defined vocabulary. Velocity token comes after each note_on
    """
    track = mid.tracks[0]
    current_tempo = 500000

    token_sequence = []
    accumulated_delta_ms = 0

    for msg in track:
        if msg.time > 0:
            delta_seconds = tick2second(msg.time, mid.ticks_per_beat, current_tempo)
            delta_ms = int(round(delta_seconds * 1000))
            accumulated_delta_ms += delta_ms

        if msg.type == "set_tempo":
            current_tempo = msg.tempo
            continue

        if msg.type == 'note_on' or msg.type == 'note_off':
            while accumulated_delta_ms > 0:
                if accumulated_delta_ms < 5:
                    accumulated_delta_ms = 0
                    break
                elif accumulated_delta_ms < 10:
                    accumulated_delta_ms = 0
                    break
                elif accumulated_delta_ms >= 1000:
                    shift = 1000
                else:
                    shift = (accumulated_delta_ms // 10) * 10
                token_sequence.append(f"TIME_SHIFT_{shift}ms")
                accumulated_delta_ms -= shift

            if msg.type == 'note_on':
                if msg.velocity == 0:
                    token_sequence.append(f"NOTE_OFF_{msg.note}")
                else:
                    token_sequence.append(f"NOTE_ON_{msg.note}")
                    velocity_bin = max(1, min(32, int(round((msg.velocity / 127) * 32))))
                    token_sequence.append(f"VELOCITY_{velocity_bin}")
            else:
                token_sequence.append(f"NOTE_OFF_{msg.note}")

    token_id_sequence = [basic_vocab[t] for t in token_sequence]

    # Trim leading/trailing time shifts
    start = 0
    while start < len(token_id_sequence) and basic_id_to_token[token_id_sequence[start]].startswith("TIME_SHIFT_"):
        start += 1
    end = len(token_id_sequence)
    while end > start and basic_id_to_token[token_id_sequence[end-1]].startswith("TIME_SHIFT_"):
        end -= 1
    token_id_sequence = token_id_sequence[start:end]

    return token_id_sequence


def tokenize_basic_vocab_velocity_bins(mid):
    """
    Velocity token applies to all subsequent notes
    """
    track = mid.tracks[0]
    current_tempo = 500000
    token_sequence = []
    accumulated_delta_ms = 0
    current_velocity = None  # Tracks the current velocity state

    for msg in track:
        if msg.time > 0:
            delta_seconds = tick2second(msg.time, mid.ticks_per_beat, current_tempo)
            delta_ms = int(round(delta_seconds * 1000))
            accumulated_delta_ms += delta_ms

        if msg.type == "set_tempo":
            current_tempo = msg.tempo
            continue

        if msg.type == 'note_on' or msg.type == 'note_off':
            # Process time shifts first
            while accumulated_delta_ms > 0:
                if accumulated_delta_ms < 5:
                    accumulated_delta_ms = 0
                    break
                elif accumulated_delta_ms < 10:
                    accumulated_delta_ms = 0
                    break
                elif accumulated_delta_ms >= 1000:
                    shift = 1000
                else:
                    shift = (accumulated_delta_ms // 10) * 10
                token_sequence.append(f"TIME_SHIFT_{shift}ms")
                accumulated_delta_ms -= shift

            if msg.type == 'note_on':
                if msg.velocity == 0:
                    token_sequence.append(f"NOTE_OFF_{msg.note}")
                else:
                    velocity_bin = max(1, min(32, int(round((msg.velocity / 127) * 32))))
                    if current_velocity != velocity_bin:
                        token_sequence.append(f"VELOCITY_{velocity_bin}")
                        current_velocity = velocity_bin
                    token_sequence.append(f"NOTE_ON_{msg.note}")
            else:
                token_sequence.append(f"NOTE_OFF_{msg.note}")

    token_id_sequence = [basic_vocab[t] for t in token_sequence]

    # Trim leading/trailing time shifts
    start = 0
    while start < len(token_id_sequence) and basic_id_to_token[token_id_sequence[start]].startswith("TIME_SHIFT_"):
        start += 1
    end = len(token_id_sequence)
    while end > start and basic_id_to_token[token_id_sequence[end-1]].startswith("TIME_SHIFT_"):
        end -= 1
    token_id_sequence = token_id_sequence[start:end]

    return token_id_sequence


def tokenize_multi_instr_vocab(mid):
    """
    Velocity token applies to all subsequent notes
    """

    def clean_tokens(sequence):
        new_sequence = []
        subseq = defaultdict(list)
        curr_instr = None

        for token in sequence:
            if "INSTRUMENT" in token:
                curr_instr = token

            elif "TIME_SHIFT" in token:
                if subseq:
                    for instr in subseq:
                        if subseq[instr]:
                            new_sequence.append(instr)
                            new_sequence.extend(subseq[instr])
                    subseq = defaultdict(list)

                new_sequence.append(token)

            else:
                if curr_instr is not None:
                    subseq[curr_instr].append(token)
                else:
                    pass
        if subseq:
            for instr in subseq:
                if subseq[instr]:
                    new_sequence.append(instr)
                    new_sequence.extend(subseq[instr])

        return new_sequence
    

    DRUM_CHANNEL = 9
    
    track = mid.tracks[0]
    current_tempo = 500000
    token_sequence = []
    accumulated_delta_ms = 0
    current_velocity = None
    current_instrument = None
    
    channels = defaultdict(int)

    for msg in track:
        if msg.time > 0:
            delta_seconds = tick2second(msg.time, mid.ticks_per_beat, current_tempo)
            delta_ms = int(round(delta_seconds * 1000))
            accumulated_delta_ms += delta_ms

        if msg.type == "set_tempo":
            current_tempo = msg.tempo
            continue

        if msg.type == "program_change":
            # Force channel 9 to use instrument 0 for drums
            if msg.channel == DRUM_CHANNEL:
                channels[msg.channel] = 0
            else:
                channels[msg.channel] = msg.program
            continue

        if msg.type in ('note_on', 'note_off'):
            # Process time shifts first
            while accumulated_delta_ms > 0:
                if accumulated_delta_ms < 5:
                    accumulated_delta_ms = 0
                    break
                elif accumulated_delta_ms < 10:
                    accumulated_delta_ms = 0
                    break
                elif accumulated_delta_ms >= 1000:
                    shift = 1000
                else:
                    shift = (accumulated_delta_ms // 10) * 10
                token_sequence.append(f"TIME_SHIFT_{shift}ms")
                accumulated_delta_ms -= shift

            if current_instrument != channels[msg.channel]:
                current_instrument = channels[msg.channel]
                token_sequence.append(f"INSTRUMENT_{current_instrument}")

            if msg.type == 'note_on':
                if msg.velocity == 0:
                    token_sequence.append(f"NOTE_OFF_{msg.note}")
                else:
                    velocity_bin = max(1, min(32, int(round((msg.velocity / 127) * 32))))
                    if current_velocity != velocity_bin:
                        token_sequence.append(f"VELOCITY_{velocity_bin}")
                        current_velocity = velocity_bin
                    token_sequence.append(f"NOTE_ON_{msg.note}")
            else:
                token_sequence.append(f"NOTE_OFF_{msg.note}")

    token_sequence = clean_tokens(token_sequence)
    
    token_id_sequence = [multi_instr_vocab[t] for t in token_sequence]

    # Trim leading/trailing time shifts
    start = 0
    while start < len(token_id_sequence) and multi_instr_id_to_token[token_id_sequence[start]].startswith("TIME_SHIFT_"):
        start += 1
    end = len(token_id_sequence)
    while end > start and multi_instr_id_to_token[token_id_sequence[end - 1]].startswith("TIME_SHIFT_"):
        end -= 1
    token_id_sequence = token_id_sequence[start:end]

    return token_id_sequence