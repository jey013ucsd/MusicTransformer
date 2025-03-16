import matplotlib.pyplot as plt
import numpy as np

def plot_midi(tokens, max_tokens=2048, title="Piano-roll Plot", scale=1.0, y_range=None, x_padding=10, color='red'):
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    
    current_time = 0
    active_notes = {}
    note_segments = []

    for token in tokens:
        if token.startswith("NOTE_ON_"):
            pitch = int(token.split("_")[-1])
            if pitch not in active_notes:
                active_notes[pitch] = current_time
        
        elif token.startswith("NOTE_OFF_"):
            pitch = int(token.split("_")[-1])
            if pitch in active_notes:
                start_time = active_notes[pitch] * scale
                end_time = current_time * scale
                note_segments.append((start_time, end_time, pitch))
                del active_notes[pitch]
        
        elif token.startswith("TIME_SHIFT_"):
            time_shift_str = token.split("_")[-1]
            shift_ms = int(time_shift_str.replace("ms", ""))
            current_time += shift_ms
    
    note_segments = np.array(note_segments)
    
    if len(note_segments) == 0:
        print("No notes found")
        return
    
    width = max(1, (current_time * scale) / 100)
    
    plt.figure(figsize=(width, 6))
    #plt.title(title)

    
    if y_range is not None:
        plt.ylim(y_range)

    plt.xlim(0 - x_padding, current_time * scale + x_padding)
    
    for seg in note_segments:
        start_t, end_t, pitch = seg
        plt.hlines(y=pitch, xmin=start_t, xmax=end_t, linewidth=4, color=color)
    
    plt.grid(True, linewidth=0.5)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    plt.show()