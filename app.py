import streamlit as st
import numpy as np
import random
from io import BytesIO
import struct
import math

# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
st.set_page_config(
    page_title="lofi machine",
    page_icon="🌙",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════
# SESSION STATE DEFAULTS
# ═══════════════════════════════════════
if "mood" not in st.session_state:
    st.session_state.mood = "Sunday Morning"
if "bpm" not in st.session_state:
    st.session_state.bpm = 78
if "bars" not in st.session_state:
    st.session_state.bars = 8
if "density" not in st.session_state:
    st.session_state.density = 0.60
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.35
if "seed_mode" not in st.session_state:
    st.session_state.seed_mode = "Random"
if "seed_value" not in st.session_state:
    st.session_state.seed_value = 42

# ═══════════════════════════════════════
# CSS
# ═══════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap');
:root {
    --bg:#000;--sf:#0a0a0a;--sf2:#111;--bd:#222;--bdl:#2a2a2a;
    --t1:#f5f5f7;--t2:#a1a1a6;--t3:#6e6e73;--t4:#48484a;
    --adim:rgba(255,255,255,.06);--r:14px;
    --font:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
    --mono:SFMono-Regular,Menlo,monospace;
}
.stApp{background:var(--bg)!important;font-family:var(--font)!important}
header[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:680px!important;padding-top:4rem!important}
section[data-testid="stSidebar"]{display:none!important}
.site-title{font-family:var(--font);font-weight:200;font-size:2.8rem;color:var(--t1);letter-spacing:-1.5px;line-height:1;margin:0;text-align:center}
.site-sub{font-family:var(--font);font-weight:300;font-size:1.05rem;color:var(--t3);text-align:center;margin-top:8px;letter-spacing:-.2px}
.sep{border:none;border-top:1px solid var(--bd);margin:2.4rem 0}
.sep-light{border:none;border-top:1px solid var(--bd);margin:1.4rem 0}
.section-label{font-family:var(--font);font-weight:500;font-size:.7rem;color:var(--t3);text-transform:uppercase;letter-spacing:1.2px;margin-bottom:16px}
.hint{font-family:var(--font);font-weight:300;font-size:.78rem;color:var(--t4);line-height:1.5;margin-top:-4px;margin-bottom:18px}
.stButton>button{font-family:var(--font)!important;font-weight:400!important;font-size:.95rem!important;letter-spacing:-.2px!important;border-radius:var(--r)!important;padding:.7rem 1.6rem!important;transition:all .2s cubic-bezier(.4,0,.2,1)!important;border:none!important}
div[data-testid="stColumns"]>div:first-child .stButton>button{background:var(--t1)!important;color:var(--bg)!important}
div[data-testid="stColumns"]>div:first-child .stButton>button:hover{background:#e0e0e0!important}
div[data-testid="stColumns"]>div:nth-child(2) .stButton>button{background:transparent!important;color:var(--t2)!important;border:1px solid var(--bdl)!important}
div[data-testid="stColumns"]>div:nth-child(2) .stButton>button:hover{background:var(--adim)!important;color:var(--t1)!important;border-color:var(--t4)!important}
.stDownloadButton>button{background:transparent!important;color:var(--t2)!important;border:1px solid var(--bdl)!important;font-family:var(--font)!important;font-weight:400!important;font-size:.85rem!important;border-radius:var(--r)!important}
.stDownloadButton>button:hover{background:var(--adim)!important;color:var(--t1)!important}
.stSelectbox label, .stRadio label, .stNumberInput label, .stSlider label{font-family:var(--font)!important;font-weight:400!important;font-size:.85rem!important;color:var(--t2)!important}
.stSelectbox>div>div, .stNumberInput>div>div>input{background:var(--sf2)!important;border:1px solid var(--bd)!important;border-radius:10px!important;color:var(--t1)!important}
.stAudio>div{border-radius:var(--r)!important}
.gen-info{display:inline-block;font-family:var(--mono);font-size:.72rem;color:var(--t4);background:var(--sf2);border:1px solid var(--bd);border-radius:8px;padding:4px 10px;margin-top:12px}
.mcmc-stats{font-family:var(--mono);font-size:.7rem;color:var(--t4);background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:14px 16px;margin-top:16px;line-height:1.8}
.empty-state{text-align:center;padding:5rem 2rem;margin-top:1rem}
.empty-icon{font-size:2.4rem;margin-bottom:1rem;opacity:.3}
.empty-text{font-family:var(--font);font-weight:300;font-size:1.1rem;color:var(--t4);letter-spacing:-.3px}
.empty-hint{font-family:var(--font);font-weight:300;font-size:.82rem;color:var(--t4);opacity:.5;margin-top:6px}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# AUDIO HELPERS
# ═══════════════════════════════════════
def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12))

def adsr(n, a=0.01, d=0.08, sus=0.7, r=0.12):
    ai, di, ri = int(n * a), int(n * d), int(n * r)
    si = max(0, n - ai - di - ri)
    env = np.zeros(n, dtype=np.float32)

    idx = 0
    if ai > 0:
        env[idx:idx + ai] = np.linspace(0, 1, ai, endpoint=False)
        idx += ai
    if di > 0:
        env[idx:idx + di] = np.linspace(1, sus, di, endpoint=False)
        idx += di
    if si > 0:
        env[idx:idx + si] = sus
        idx += si
    if idx < n:
        env[idx:] = np.linspace(sus, 0, n - idx, endpoint=False)

    return env

def soft_synth(freq, dur, sr=44100, detune=6):
    n = max(1, int(dur * sr))
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
    f2 = freq * (2 ** (detune / 1200))

    wave = (
        0.50 * np.sin(2 * np.pi * freq * t) +
        0.22 * np.sin(2 * np.pi * f2 * t) +
        0.18 * np.sin(2 * np.pi * 2 * freq * t) +
        0.08 * np.sin(2 * np.pi * 0.5 * freq * t)
    ).astype(np.float32)

    wave = np.convolve(wave, np.ones(8, dtype=np.float32) / 8, mode="same")
    return wave * adsr(n, a=0.02, d=0.08, sus=0.65, r=0.18)

def bass_synth(freq, dur, sr=44100):
    n = max(1, int(dur * sr))
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)

    wave = (
        0.70 * np.sin(2 * np.pi * freq * t) +
        0.20 * np.sin(2 * np.pi * 2 * freq * t)
    ).astype(np.float32)

    wave = np.convolve(wave, np.ones(12, dtype=np.float32) / 12, mode="same")
    return wave * adsr(n, a=0.005, d=0.08, sus=0.75, r=0.10)

def kick_sound(sr=44100):
    dur = 0.18
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
    return (0.9 * np.sin(2 * np.pi * (150 * np.exp(-22 * t) + 35) * t) * np.exp(-18 * t)).astype(np.float32)

def snare_sound(sr=44100):
    dur = 0.12
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
    s = 0.75 * np.random.normal(0, 1, n).astype(np.float32) + 0.25 * np.sin(2 * np.pi * 180 * t)
    s = np.convolve(s, np.ones(6, dtype=np.float32) / 6, mode="same")
    return (0.35 * s * np.exp(-28 * t)).astype(np.float32)

def hihat_sound(sr=44100):
    dur = 0.05
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False, dtype=np.float32)
    return (0.10 * np.random.uniform(-1, 1, n).astype(np.float32) * np.exp(-70 * t)).astype(np.float32)

def add_feedback_delay(sig, dt=0.33, fb=0.22, sr=44100):
    delay_samples = int(dt * sr)
    if delay_samples <= 0:
        return sig.copy()
    out = sig.copy()
    for i in range(delay_samples, len(sig)):
        out[i] += fb * out[i - delay_samples]
    return out

def pan_stereo(mono, p):
    p = max(-1.0, min(1.0, p))
    left = mono * np.sqrt((1 - p) / 2)
    right = mono * np.sqrt((1 + p) / 2)
    return np.column_stack([left, right]).astype(np.float32)

def place_wave(track, start, wave, gain=1.0):
    if start >= len(track):
        return
    end = min(start + len(wave), len(track))
    if end > start:
        track[start:end] += gain * wave[:end - start]

# ═══════════════════════════════════════
# MCMC ENERGY FUNCTIONS
# ═══════════════════════════════════════
def melody_energy(note, prev_note, chord_notes, scale_notes, temperature):
    e = 0.0

    chord_set = set(n % 12 for n in chord_notes)
    if note % 12 in chord_set:
        e += 0.0
    else:
        min_dist = min(min(abs(note % 12 - c), 12 - abs(note % 12 - c)) for c in chord_set)
        e += min_dist * 1.5

    scale_set = set(n % 12 for n in scale_notes)
    if note % 12 not in scale_set:
        e += 3.0

    if prev_note is not None:
        interval = abs(note - prev_note)

        # Higher temperature allows wider melodic exploration
        jump_scale = max(0.35, 1.0 - 0.55 * min(temperature / 3.3, 1.0))

        if interval == 0:
            e += 0.8
        elif interval <= 2:
            e += 0.0
        elif interval <= 4:
            e += 0.3
        elif interval <= 7:
            e += 1.0 * jump_scale
        else:
            e += interval * 0.4 * jump_scale

    if note < 55 or note > 88:
        e += 4.0
    elif note < 60 or note > 84:
        e += 1.0

    return e

def bass_energy(note, root, prev_bass, chord_notes):
    e = 0.0

    if note % 12 == root % 12:
        e += 0.0
    elif note % 12 == (root + 7) % 12:
        e += 0.5
    elif note % 12 in set(n % 12 for n in chord_notes):
        e += 1.5
    else:
        e += 5.0

    if prev_bass is not None:
        interval = abs(note - prev_bass)
        if interval <= 2:
            e += 0.0
        elif interval <= 5:
            e += 0.5
        elif interval <= 7:
            e += 1.0
        else:
            e += interval * 0.3

    if note < 33 or note > 60:
        e += 5.0
    elif note < 36 or note > 55:
        e += 1.5

    return e

def kick_rhythm_energy(pattern, base_pattern, density_target):
    e = 0.0

    for i in base_pattern:
        if i < len(pattern) and not pattern[i]:
            e += 3.0

    actual_density = sum(pattern) / len(pattern)
    e += abs(actual_density - density_target) * 7.0

    on_offbeat = sum(1 for i, hit in enumerate(pattern) if hit and i % 2 == 1)
    e -= on_offbeat * 0.35

    for i in range(len(pattern) - 1):
        if pattern[i] and pattern[i + 1]:
            e += 0.25

    return e

def snare_rhythm_energy(pattern, base_pattern, density_target):
    e = 0.0

    for i in base_pattern:
        if i < len(pattern) and not pattern[i]:
            e += 4.0

    actual_density = sum(pattern) / len(pattern)
    e += abs(actual_density - density_target) * 9.0

    for i in range(len(pattern) - 1):
        if pattern[i] and pattern[i + 1]:
            e += 1.2

    return e

def voicing_energy(voicing, prev_voicing):
    e = 0.0

    spread = max(voicing) - min(voicing)
    if spread < 7:
        e += 3.0
    elif spread < 10:
        e += 1.0
    elif spread > 24:
        e += 2.0
    elif 12 <= spread <= 20:
        e -= 0.5

    if prev_voicing is not None and len(prev_voicing) == len(voicing):
        total_motion = sum(abs(a - b) for a, b in zip(voicing, prev_voicing))
        e += total_motion * 0.15

    for n in voicing:
        if n < 36 or n > 84:
            e += 2.0
        elif n < 42 or n > 78:
            e += 0.5

    sv = sorted(voicing)
    for i in range(len(sv) - 1):
        gap = sv[i + 1] - sv[i]
        if gap < 2:
            e += 2.0
        elif gap < 3:
            e += 0.5

    # Penalize voice crossing relative to sorted order target
    if voicing != sorted(voicing):
        e += 1.5

    return e

# ═══════════════════════════════════════
# MH SAMPLER
# ═══════════════════════════════════════
def mh_sample(current_state, energy_fn, proposal_fn, temperature, n_steps=20):
    state = current_state
    current_e = energy_fn(state)
    accepted = 0
    T = max(temperature, 0.01)

    for _ in range(n_steps):
        proposal = proposal_fn(state)
        proposal_e = energy_fn(proposal)
        delta = proposal_e - current_e

        if delta <= 0:
            state = proposal
            current_e = proposal_e
            accepted += 1
        else:
            accept_prob = math.exp(-delta / T)
            if random.random() < accept_prob:
                state = proposal
                current_e = proposal_e
                accepted += 1

    return state, accepted

# ═══════════════════════════════════════
# PROPOSALS
# ═══════════════════════════════════════
def propose_melody_note(current_note):
    step = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
    return current_note + step

def propose_bass_note(current_note):
    step = random.choice([-7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7])
    return current_note + step

def propose_rhythm_flip(pattern):
    new = list(pattern)
    i = random.randint(0, len(new) - 1)
    new[i] = not new[i]
    return new

def propose_voicing(voicing):
    new = list(voicing)
    i = random.randint(0, len(new) - 1)
    new[i] += random.choice([-3, -2, -1, 1, 2, 3])

    # keep voices sorted to preserve identity a bit better
    new = sorted(new)
    return new

# ═══════════════════════════════════════
# MOODS
# ═══════════════════════════════════════
MOODS = {
    "Rainy Window": {
        "description": "Melancholic chords, sparse melody, slow swing",
        "bpm_range": (62, 74),
        "scale": [57, 60, 62, 64, 67, 69],
        "chords": [[45, 52, 57, 60], [41, 48, 53, 57], [48, 52, 55, 60], [43, 47, 50, 55]],
        "progression": [0, 1, 2, 3, 0, 1, 2, 3],
        "base_kicks": [0, 8],
        "base_snares": [4, 12],
        "swing": 0.08,
        "melody_density": 0.55,
        "vinyl": 0.020,
        "pad": 0.060,
    },
    "Sunday Morning": {
        "description": "Warm major chords, playful melody, gentle groove",
        "bpm_range": (72, 84),
        "scale": [60, 62, 64, 67, 69, 72],
        "chords": [[48, 55, 60, 64], [45, 52, 57, 60], [41, 48, 52, 57], [43, 50, 55, 59]],
        "progression": [0, 1, 2, 3, 0, 1, 3, 2],
        "base_kicks": [0, 7, 8],
        "base_snares": [4, 12],
        "swing": 0.06,
        "melody_density": 0.75,
        "vinyl": 0.015,
        "pad": 0.040,
    },
    "Late Night Drive": {
        "description": "Dark jazzy voicings, deep bass, heavy swing",
        "bpm_range": (66, 78),
        "scale": [62, 64, 65, 67, 69, 71, 72],
        "chords": [[50, 57, 60, 64], [43, 50, 55, 60], [48, 55, 59, 64], [41, 48, 52, 57]],
        "progression": [0, 1, 2, 3, 2, 0, 1, 3],
        "base_kicks": [0, 3, 10],
        "base_snares": [4, 12],
        "swing": 0.10,
        "melody_density": 0.60,
        "vinyl": 0.018,
        "pad": 0.050,
    },
    "Café in Kyoto": {
        "description": "Pentatonic melody, minimal drums, lots of space",
        "bpm_range": (68, 80),
        "scale": [64, 67, 69, 71, 74, 76],
        "chords": [[40, 47, 52, 55], [48, 55, 60, 64], [45, 52, 57, 60], [47, 54, 59, 62]],
        "progression": [0, 1, 2, 3, 0, 2, 1, 3],
        "base_kicks": [0, 10],
        "base_snares": [8],
        "swing": 0.05,
        "melody_density": 0.50,
        "vinyl": 0.022,
        "pad": 0.070,
    },
    "Rooftop Sunset": {
        "description": "Bright and uplifting, open chords, airy melody",
        "bpm_range": (76, 88),
        "scale": [65, 67, 69, 71, 72, 74, 76],
        "chords": [[41, 48, 52, 57], [40, 47, 52, 55], [50, 57, 60, 65], [48, 55, 60, 64]],
        "progression": [0, 1, 2, 3, 0, 1, 3, 2],
        "base_kicks": [0, 7, 8, 11],
        "base_snares": [4, 12, 14],
        "swing": 0.04,
        "melody_density": 0.80,
        "vinyl": 0.012,
        "pad": 0.030,
    },
}

# ═══════════════════════════════════════
# GENERATOR
# ═══════════════════════════════════════
def generate_beat(mood_name, bpm, bars, density, temperature):
    sr = 44100
    mood = MOODS[mood_name]
    beat_sec = 60 / bpm
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    total_dur = total_beats * beat_sec
    n_total = int(total_dur * sr)

    melody_track = np.zeros(n_total, dtype=np.float32)
    chord_track = np.zeros(n_total, dtype=np.float32)
    bass_track = np.zeros(n_total, dtype=np.float32)
    drum_track = np.zeros(n_total, dtype=np.float32)

    chords_base = mood["chords"]
    prog = mood["progression"]
    scale = mood["scale"]
    swing = mood["swing"]

    mel_vol = 0.10 + 0.18 * density
    chd_vol = 0.06 + 0.10 * density
    bas_vol = 0.18 + 0.14 * density
    mel_dens = mood["melody_density"] * (0.5 + 0.5 * density)
    vinyl_amt = mood["vinyl"] * (0.6 + 0.4 * density)
    pad_vol = mood["pad"] * (0.4 + 0.6 * density)

    T_melody = 0.3 + temperature * 3.0
    T_bass = 0.2 + temperature * 2.0
    T_rhythm = 0.3 + temperature * 2.5
    T_voicing = 0.2 + temperature * 2.0

    burn_in = 15
    sample_steps = 20

    stats = {
        "melody_accepted": 0, "melody_total": 0,
        "bass_accepted": 0, "bass_total": 0,
        "rhythm_accepted": 0, "rhythm_total": 0,
        "voicing_accepted": 0, "voicing_total": 0,
    }

    # Precompute drum one-shots once
    kick_one = kick_sound(sr)
    snare_one = snare_sound(sr)
    hihat_one = hihat_sound(sr)

    # ─── MCMC: CHORD VOICINGS ───
    prev_voicing = None
    bar_voicings = []

    for bar in range(bars):
        base_chord = chords_base[prog[bar % len(prog)]]

        def voicing_e(v):
            return voicing_energy(v, prev_voicing)

        state = sorted(list(base_chord))
        state, _ = mh_sample(state, voicing_e, propose_voicing, T_voicing, burn_in)
        state, acc = mh_sample(state, voicing_e, propose_voicing, T_voicing, sample_steps)

        stats["voicing_accepted"] += acc
        stats["voicing_total"] += sample_steps

        base_pcs = [n % 12 for n in base_chord]
        final_voicing = []
        for i, pc in enumerate(base_pcs):
            target = state[i]
            best = target
            best_dist = 999
            for octave_shift in range(-24, 25):
                candidate = target + octave_shift
                if candidate % 12 == pc and abs(octave_shift) < best_dist:
                    best = candidate
                    best_dist = abs(octave_shift)
            final_voicing.append(best)

        final_voicing = sorted(final_voicing)
        bar_voicings.append(final_voicing)
        prev_voicing = final_voicing

    # ─── RENDER CHORDS ───
    for bar in range(bars):
        voicing = bar_voicings[bar]
        start_t = bar * beats_per_bar * beat_sec
        dur = beats_per_bar * beat_sec
        start = int(start_t * sr)

        for note in voicing:
            wave = soft_synth(midi_to_freq(note), dur, sr, detune=4)
            place_wave(chord_track, start, wave, gain=chd_vol)

    # ─── MCMC: MELODY ───
    positions = [0.0, 0.75, 1.5, 2.5, 3.25]
    prev_melody_note = random.choice(scale) + 12

    for bar in range(bars):
        voicing = bar_voicings[bar]

        for pos in positions:
            skip_threshold = 1.0 - mel_dens
            if random.random() < skip_threshold * (1.0 - temperature * 0.3):
                continue

            current = prev_melody_note

            def mel_e(note):
                return melody_energy(note, prev_melody_note, voicing, scale, T_melody)

            current, _ = mh_sample(current, mel_e, propose_melody_note, T_melody, burn_in)
            note, acc = mh_sample(current, mel_e, propose_melody_note, T_melody, sample_steps)

            stats["melody_accepted"] += acc
            stats["melody_total"] += sample_steps

            dur_choices = [0.4, 0.5, 0.75, 1.0]
            if temperature > 0.6:
                dur_choices += [0.25, 1.5]
            dur = random.choice(dur_choices) * beat_sec

            start = int((bar * beats_per_bar + pos) * beat_sec * sr)
            wave = soft_synth(midi_to_freq(note), dur, sr, detune=8)
            place_wave(melody_track, start, wave, gain=mel_vol)

            prev_melody_note = note

    # ─── MCMC: BASS ───
    prev_bass_note = None
    bass_positions = [0.0, 1.5, 2.0, 3.0]
    bass_durations = [1.0, 0.4, 0.8, 0.7]

    for bar in range(bars):
        base_chord = chords_base[prog[bar % len(prog)]]
        root = base_chord[0]

        for pos, ln in zip(bass_positions, bass_durations):
            current = root if prev_bass_note is None else prev_bass_note

            def bass_e(note):
                return bass_energy(note, root, prev_bass_note, base_chord)

            current, _ = mh_sample(current, bass_e, propose_bass_note, T_bass, burn_in)
            note, acc = mh_sample(current, bass_e, propose_bass_note, T_bass, sample_steps)

            stats["bass_accepted"] += acc
            stats["bass_total"] += sample_steps

            start = int((bar * beats_per_bar + pos) * beat_sec * sr)
            wave = bass_synth(midi_to_freq(note), ln * beat_sec, sr)
            place_wave(bass_track, start, wave, gain=bas_vol)

            prev_bass_note = note

    # ─── MCMC: DRUMS ───
    kick_density_target = len(mood["base_kicks"]) / 16 + density * 0.08
    snare_density_target = len(mood["base_snares"]) / 16 + density * 0.04
    sixteenth = beat_sec / 4

    for bar in range(bars):
        bar_time = bar * beats_per_bar * beat_sec

        init_kick = [False] * 16
        for i in mood["base_kicks"]:
            if i < 16:
                init_kick[i] = True

        def kick_e(pat):
            return kick_rhythm_energy(pat, mood["base_kicks"], kick_density_target)

        kick_state, _ = mh_sample(init_kick, kick_e, propose_rhythm_flip, T_rhythm, burn_in)
        kick_pattern, acc = mh_sample(kick_state, kick_e, propose_rhythm_flip, T_rhythm, sample_steps)
        stats["rhythm_accepted"] += acc
        stats["rhythm_total"] += sample_steps

        init_snare = [False] * 16
        for i in mood["base_snares"]:
            if i < 16:
                init_snare[i] = True

        def snare_e(pat):
            return snare_rhythm_energy(pat, mood["base_snares"], snare_density_target)

        snare_state, _ = mh_sample(init_snare, snare_e, propose_rhythm_flip, T_rhythm, burn_in)
        snare_pattern, acc = mh_sample(snare_state, snare_e, propose_rhythm_flip, T_rhythm, sample_steps)
        stats["rhythm_accepted"] += acc
        stats["rhythm_total"] += sample_steps

        for step in range(16):
            t = bar_time + step * sixteenth
            if step % 2 == 1:
                t += swing * beat_sec

            start = int(t * sr)

            place_wave(drum_track, start, hihat_one, gain=1.0)

            if kick_pattern[step]:
                place_wave(drum_track, start, kick_one, gain=0.9)

            if snare_pattern[step]:
                place_wave(drum_track, start, snare_one, gain=1.0)

    # ─── TEXTURE ───
    noise = np.random.normal(0, 1, n_total).astype(np.float32)
    noise = np.convolve(noise, np.ones(30, dtype=np.float32) / 30, mode="same")
    vinyl = vinyl_amt * noise

    t_full = np.linspace(0, total_dur, n_total, endpoint=False, dtype=np.float32)
    pad = pad_vol * (
        0.5 * np.sin(2 * np.pi * midi_to_freq(48) * t_full) +
        0.3 * np.sin(2 * np.pi * midi_to_freq(55) * t_full) +
        0.2 * np.sin(2 * np.pi * midi_to_freq(60) * t_full)
    ).astype(np.float32)

    # ─── STEREO MIX ───
    mix = (
        pan_stereo(melody_track, 0.25) +
        pan_stereo(chord_track, -0.20) +
        pan_stereo(bass_track, 0.00) +
        pan_stereo(drum_track, 0.05) +
        pan_stereo(vinyl + pad, -0.10)
    )

    left = add_feedback_delay(mix[:, 0], dt=0.36, fb=0.16, sr=sr)
    right = add_feedback_delay(mix[:, 1], dt=0.42, fb=0.14, sr=sr)

    stereo = np.column_stack([left, right]).astype(np.float32)
    stereo[:, 0] = np.convolve(stereo[:, 0], np.ones(4, dtype=np.float32) / 4, mode="same")
    stereo[:, 1] = np.convolve(stereo[:, 1], np.ones(4, dtype=np.float32) / 4, mode="same")

    peak = np.max(np.abs(stereo))
    if peak > 1e-9:
        stereo /= peak

    return stereo, sr, stats

# ═══════════════════════════════════════
# WAV EXPORT
# ═══════════════════════════════════════
def to_wav(stereo, sr):
    data = np.int16(np.clip(stereo, -1.0, 1.0) * 32767)
    buf = BytesIO()

    n_samples = data.shape[0]
    n_channels = 2
    sample_width = 2
    data_size = n_samples * n_channels * sample_width

    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sr, sr * n_channels * sample_width, n_channels * sample_width, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(data.tobytes())
    buf.seek(0)
    return buf

# ═══════════════════════════════════════
# RANDOMIZER
# ═══════════════════════════════════════
def randomize_controls():
    mood = random.choice(list(MOODS.keys()))
    lo, hi = MOODS[mood]["bpm_range"]

    st.session_state.mood = mood
    st.session_state.bpm = random.randint(lo, hi)
    st.session_state.bars = random.choice([8, 12, 16])
    st.session_state.density = round(random.uniform(0.30, 0.90), 2)
    st.session_state.temperature = round(random.uniform(0.10, 0.85), 2)

    if st.session_state.seed_mode == "Fixed":
        st.session_state.seed_value = random.randint(0, 999999)

# ═══════════════════════════════════════
# UI
# ═══════════════════════════════════════
st.markdown('<p class="site-title">lofi machine</p>', unsafe_allow_html=True)
st.markdown('<p class="site-sub">Markov Chain Monte Carlo meets lo-fi hip hop.</p>', unsafe_allow_html=True)
st.markdown('<hr class="sep">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Mood</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hint">The seed of everything. Harmonic palette, scale, rhythmic skeleton, and atmosphere all start here.</p>',
    unsafe_allow_html=True
)

mood_names = list(MOODS.keys())
mood = st.selectbox(
    "Mood",
    mood_names,
    index=mood_names.index(st.session_state.mood),
    format_func=lambda x: f"{x}  —  {MOODS[x]['description']}",
    label_visibility="collapsed",
    key="mood"
)

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

lo, hi = MOODS[mood]["bpm_range"]
st.markdown('<p class="section-label">Tempo</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="hint">Sweet spot for this mood: {lo} to {hi} BPM. The sampler still works outside that range.</p>',
    unsafe_allow_html=True
)
bpm = st.slider("BPM", 55, 110, st.session_state.bpm, 1, label_visibility="collapsed", key="bpm")

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Duration</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hint">More bars means more room for the chains to evolve and discover new local patterns.</p>',
    unsafe_allow_html=True
)
bars = st.select_slider(
    "Bars",
    options=[4, 8, 12, 16, 24, 32],
    value=st.session_state.bars,
    label_visibility="collapsed",
    key="bars"
)

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Density</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hint">Controls how full the arrangement feels, including note frequency, texture amount, and layer balance.</p>',
    unsafe_allow_html=True
)
density = st.slider(
    "Density", 0.0, 1.0, st.session_state.density, 0.05,
    label_visibility="collapsed", key="density"
)

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Temperature</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hint">Low temperature makes the sampler conservative and consonant. High temperature increases exploration, larger intervals, and stranger rhythmic outcomes.</p>',
    unsafe_allow_html=True
)
temperature = st.slider(
    "Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
    label_visibility="collapsed", key="temperature"
)

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

st.markdown('<p class="section-label">Reproducibility</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hint">Use Random for a fresh result every time, or Fixed to reproduce the same track from the same controls and seed.</p>',
    unsafe_allow_html=True
)

seed_mode = st.radio(
    "Seed mode",
    options=["Random", "Fixed"],
    horizontal=True,
    label_visibility="collapsed",
    key="seed_mode"
)

seed_value = None
if seed_mode == "Fixed":
    seed_value = st.number_input(
        "Seed",
        min_value=0,
        max_value=999999999,
        value=st.session_state.seed_value,
        step=1,
        key="seed_value"
    )

st.markdown('<hr class="sep">', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    generate_btn = st.button("Generate", use_container_width=True)
with col2:
    random_btn = st.button("Trust the Randomness", use_container_width=True)

if random_btn:
    randomize_controls()
    st.rerun()

if generate_btn:
    if seed_mode == "Fixed":
        random.seed(int(seed_value))
        np.random.seed(int(seed_value))
        used_seed = int(seed_value)
    else:
        used_seed = random.randint(0, 999999999)
        random.seed(used_seed)
        np.random.seed(used_seed)

    with st.spinner(""):
        stereo, sr, stats = generate_beat(
            mood_name=st.session_state.mood,
            bpm=st.session_state.bpm,
            bars=st.session_state.bars,
            density=st.session_state.density,
            temperature=st.session_state.temperature
        )
        wav = to_wav(stereo, sr)

    dur_sec = round(st.session_state.bars * 4 * (60 / st.session_state.bpm), 1)

    def rate(a, t):
        return f"{(100 * a / t):.0f}%" if t > 0 else "—"

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.audio(wav, format="audio/wav")
    st.markdown(
        f'<p class="gen-info">{st.session_state.mood} · {st.session_state.bpm} bpm · {st.session_state.bars} bars · {dur_sec}s · T={st.session_state.temperature} · seed={used_seed}</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""<div class="mcmc-stats">
<strong style="color:#6e6e73">MCMC Diagnostics</strong><br>
Melody chain &nbsp;→&nbsp; {stats["melody_accepted"]}/{stats["melody_total"]} accepted ({rate(stats["melody_accepted"], stats["melody_total"])})<br>
Bass chain &nbsp;&nbsp;&nbsp;→&nbsp; {stats["bass_accepted"]}/{stats["bass_total"]} accepted ({rate(stats["bass_accepted"], stats["bass_total"])})<br>
Rhythm chain &nbsp;→&nbsp; {stats["rhythm_accepted"]}/{stats["rhythm_total"]} accepted ({rate(stats["rhythm_accepted"], stats["rhythm_total"])})<br>
Voicing chain →&nbsp; {stats["voicing_accepted"]}/{stats["voicing_total"]} accepted ({rate(stats["voicing_accepted"], stats["voicing_total"])})
</div>""",
        unsafe_allow_html=True
    )

    st.download_button(
        "Download WAV",
        wav,
        file_name=f"lofi_mcmc_{st.session_state.mood.lower().replace(' ', '_')}_{st.session_state.bpm}bpm_T{st.session_state.temperature}_seed{used_seed}.wav",
        mime="audio/wav",
        use_container_width=True,
    )
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎧</div>
        <p class="empty-text">Four Markov chains, waiting to compose.</p>
        <p class="empty-hint">Or let randomness decide everything.</p>
    </div>
    """, unsafe_allow_html=True)
