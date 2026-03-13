import streamlit as st
import numpy as np
import random
from io import BytesIO
import struct
import base64

# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="lofi machine",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================
# CUSTOM CSS
# ======================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Outfit:wght@200;300;400;500;600&display=swap');

/* Root variables */
:root {
    --bg-deep: #0d0d0f;
    --bg-card: #161619;
    --bg-card-hover: #1c1c20;
    --border-subtle: #2a2a30;
    --text-primary: #e8e4df;
    --text-secondary: #8a8680;
    --text-muted: #5a5854;
    --accent-warm: #d4926a;
    --accent-cool: #6a8fd4;
    --accent-glow: #d4926a22;
    --radius: 12px;
}

/* Global overrides */
.stApp {
    background-color: var(--bg-deep) !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111113 !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label {
    color: var(--text-secondary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-warm) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 8px;
    margin-top: 1.5rem;
}

/* Sliders */
.stSlider > div > div > div > div {
    background-color: var(--accent-warm) !important;
}

.stSlider label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
}

/* Select boxes */
.stSelectbox label, .stMultiSelect label {
    font-family: 'DM Mono', monospace !important;
    color: var(--text-muted) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-warm), #c47a52) !important;
    color: #0d0d0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 1px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px var(--accent-glow) !important;
}

/* Audio player */
.stAudio {
    border-radius: var(--radius) !important;
    overflow: hidden;
}

/* Title area */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 200;
    font-size: 3.2rem;
    color: var(--text-primary);
    letter-spacing: -1px;
    margin-bottom: 0;
    line-height: 1.1;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}

.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.status-ready {
    background: #1a2e1a;
    color: #6ad46a;
    border: 1px solid #2a4a2a;
}

.divider-line {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 1.5rem 0;
}

.param-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}

/* Expanders */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 1px solid var(--border-subtle);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.stTabs [aria-selected="true"] {
    color: var(--accent-warm) !important;
    border-bottom-color: var(--accent-warm) !important;
}

/* Checkbox */
.stCheckbox label span {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
}

/* Number input */
.stNumberInput label {
    font-family: 'DM Mono', monospace !important;
    color: var(--text-muted) !important;
}
</style>
""", unsafe_allow_html=True)


# ======================================
# AUDIO ENGINE
# ======================================
def midi_to_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12))


def adsr(n, attack=0.01, decay=0.08, sustain=0.7, release=0.12):
    a = int(n * attack)
    d = int(n * decay)
    r = int(n * release)
    s = max(0, n - a - d - r)
    env = np.zeros(n)
    if a > 0:
        env[:a] = np.linspace(0, 1, a, endpoint=False)
    if d > 0:
        env[a:a+d] = np.linspace(1, sustain, d, endpoint=False)
    if s > 0:
        env[a+d:a+d+s] = sustain
    if r > 0:
        env[a+d+s:] = np.linspace(sustain, 0, n - (a+d+s), endpoint=False)
    return env


def soft_synth(freq, dur, sr=44100, detune_cents=6):
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False)
    detune_ratio = 2 ** (detune_cents / 1200)
    f2 = freq * detune_ratio
    wave = (
        0.50 * np.sin(2 * np.pi * freq * t) +
        0.22 * np.sin(2 * np.pi * f2 * t) +
        0.18 * np.sin(2 * np.pi * 2 * freq * t) +
        0.08 * np.sin(2 * np.pi * 0.5 * freq * t)
    )
    wave = np.convolve(wave, np.ones(8) / 8, mode="same")
    env = adsr(n, attack=0.02, decay=0.08, sustain=0.65, release=0.18)
    return wave * env


def bass_synth(freq, dur, sr=44100):
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (
        0.7 * np.sin(2 * np.pi * freq * t) +
        0.2 * np.sin(2 * np.pi * 2 * freq * t)
    )
    env = adsr(n, attack=0.005, decay=0.08, sustain=0.75, release=0.1)
    wave = np.convolve(wave, np.ones(12) / 12, mode="same")
    return wave * env


def kick(sr=44100, dur=0.18):
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False)
    freq = 150 * np.exp(-22 * t) + 35
    wave = np.sin(2 * np.pi * freq * t)
    env = np.exp(-18 * t)
    return 0.9 * wave * env


def snare(sr=44100, dur=0.12):
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False)
    noise = np.random.normal(0, 1, n)
    tone = np.sin(2 * np.pi * 180 * t)
    env = np.exp(-28 * t)
    sig = 0.75 * noise + 0.25 * tone
    sig = np.convolve(sig, np.ones(6) / 6, mode="same")
    return 0.35 * sig * env


def hihat(sr=44100, dur=0.05):
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False)
    noise = np.random.uniform(-1, 1, n)
    env = np.exp(-70 * t)
    return 0.10 * noise * env


def add_delay(signal, delay_time=0.33, feedback=0.22, sr=44100):
    delay_samples = int(delay_time * sr)
    out = np.copy(signal)
    for i in range(delay_samples, len(signal)):
        out[i] += feedback * out[i - delay_samples]
    return out


def pan_stereo(mono, pan):
    left = mono * np.sqrt((1 - pan) / 2)
    right = mono * np.sqrt((1 + pan) / 2)
    return np.column_stack([left, right])


# ======================================
# SCALE / CHORD DEFINITIONS
# ======================================
SCALE_PRESETS = {
    "C Major Pentatonic": [60, 62, 64, 67, 69, 72],
    "A Minor Pentatonic": [57, 60, 62, 64, 67, 69],
    "D Dorian": [62, 64, 65, 67, 69, 71, 72],
    "F Lydian": [65, 67, 69, 71, 72, 74, 76],
    "G Mixolydian": [55, 57, 59, 60, 62, 64, 65],
    "E Minor": [64, 66, 67, 69, 71, 72, 74],
    "Bb Major": [58, 60, 62, 63, 65, 67, 69],
}

CHORD_PRESETS = {
    "Dreamy (Cmaj7 → Am7 → Fmaj7 → G7)": {
        "chords": [
            [48, 55, 60, 64],
            [45, 52, 57, 60],
            [41, 48, 52, 57],
            [43, 50, 55, 59],
        ],
        "progression": [0, 1, 2, 3, 0, 1, 3, 2],
    },
    "Melancholy (Am → F → C → G)": {
        "chords": [
            [45, 52, 57, 60],
            [41, 48, 53, 57],
            [48, 52, 55, 60],
            [43, 47, 50, 55],
        ],
        "progression": [0, 1, 2, 3, 0, 1, 2, 3],
    },
    "Jazzy (Dm9 → G13 → Cmaj9 → Fmaj7)": {
        "chords": [
            [50, 57, 60, 64],
            [43, 50, 55, 60],
            [48, 55, 59, 64],
            [41, 48, 52, 57],
        ],
        "progression": [0, 1, 2, 3, 2, 0, 1, 3],
    },
    "Chill Night (Em7 → Cmaj7 → Am7 → Bm7)": {
        "chords": [
            [40, 47, 52, 55],
            [48, 55, 60, 64],
            [45, 52, 57, 60],
            [47, 54, 59, 62],
        ],
        "progression": [0, 1, 2, 3, 0, 2, 1, 3],
    },
    "Rainy Day (Fmaj7 → Em7 → Dm7 → Cmaj7)": {
        "chords": [
            [41, 48, 52, 57],
            [40, 47, 52, 55],
            [50, 57, 60, 65],
            [48, 55, 60, 64],
        ],
        "progression": [0, 1, 2, 3, 0, 1, 3, 2],
    },
}

DRUM_PATTERNS = {
    "Classic Boom-Bap": {
        "kick_pos": [0.0, 1.75, 2.0],
        "snare_pos": [1.0, 3.0],
    },
    "Laid Back": {
        "kick_pos": [0.0, 2.5],
        "snare_pos": [1.0, 3.0],
    },
    "Busy Fingers": {
        "kick_pos": [0.0, 0.75, 2.0, 2.75],
        "snare_pos": [1.0, 3.0, 3.5],
    },
    "Minimal Pulse": {
        "kick_pos": [0.0, 2.0],
        "snare_pos": [2.0],
    },
    "Syncopated": {
        "kick_pos": [0.0, 1.25, 2.5, 3.75],
        "snare_pos": [1.0, 3.0],
    },
}


# ======================================
# MAIN GENERATOR
# ======================================
def generate_beat(params):
    sr = 44100
    bpm = params["bpm"]
    bars = params["bars"]
    beat_sec = 60 / bpm
    beats_per_bar = 4
    total_beats = bars * beats_per_bar
    total_duration = total_beats * beat_sec
    n_total = int(total_duration * sr)

    # seed
    if params["seed"] is not None:
        random.seed(params["seed"])
        np.random.seed(params["seed"])

    # tracks
    melody_track = np.zeros(n_total)
    chord_track = np.zeros(n_total)
    bass_track = np.zeros(n_total)
    drum_track = np.zeros(n_total)

    # resolve chords & progression
    chord_data = CHORD_PRESETS[params["chord_preset"]]
    chords = chord_data["chords"]
    base_prog = chord_data["progression"]
    # extend or trim progression to match bars
    progression = []
    for i in range(bars):
        progression.append(base_prog[i % len(base_prog)])

    scale = SCALE_PRESETS[params["scale"]]

    # ---- CHORD LAYER ----
    if params["enable_chords"]:
        for bar in range(bars):
            chord = chords[progression[bar]]
            start_t = bar * beats_per_bar * beat_sec
            dur = beats_per_bar * beat_sec
            for note in chord:
                freq = midi_to_freq(note)
                wave = soft_synth(freq, dur, sr, detune_cents=params["chord_detune"])
                start = int(start_t * sr)
                end = start + len(wave)
                if end <= n_total:
                    chord_track[start:end] += params["chord_vol"] * wave

    # ---- MELODY ----
    if params["enable_melody"]:
        for bar in range(bars):
            chord = chords[progression[bar]]
            chord_tones_upper = [n + 12 for n in chord[1:]]
            beat_positions = [0.0, 0.75, 1.5, 2.5, 3.25]
            last_note = random.choice(chord_tones_upper)

            for pos in beat_positions:
                if random.random() < (1 - params["melody_density"]):
                    continue
                note_pool = chord_tones_upper + scale
                next_candidates = [n for n in note_pool if abs(n - last_note) <= params["melody_range"]]
                if not next_candidates:
                    next_candidates = note_pool
                pitch = random.choice(next_candidates)
                dur_beats = random.choice([0.4, 0.5, 0.75, 1.0])
                dur = dur_beats * beat_sec
                start_t = (bar * beats_per_bar + pos) * beat_sec
                start = int(start_t * sr)
                wave = soft_synth(midi_to_freq(pitch), dur, sr, detune_cents=params["melody_detune"])
                end = min(start + len(wave), n_total)
                melody_track[start:end] += params["melody_vol"] * wave[:end - start]
                last_note = pitch

    # ---- BASS ----
    if params["enable_bass"]:
        for bar in range(bars):
            chord = chords[progression[bar]]
            root = chord[0]
            pattern = [0, 1.5, 2.0, 3.0]
            lengths = [1.0, 0.4, 0.8, 0.7]
            for pos, ln in zip(pattern, lengths):
                start_t = (bar * beats_per_bar + pos) * beat_sec
                dur = ln * beat_sec
                start = int(start_t * sr)
                pitch = root
                if random.random() < params["bass_fifth_chance"]:
                    pitch = root + 7
                wave = bass_synth(midi_to_freq(pitch), dur, sr)
                end = min(start + len(wave), n_total)
                bass_track[start:end] += params["bass_vol"] * wave[:end - start]

    # ---- DRUMS ----
    if params["enable_drums"]:
        drum_pat = DRUM_PATTERNS[params["drum_pattern"]]
        swing = params["swing"] * beat_sec

        # hi-hats
        if params["enable_hihats"]:
            for beat in range(total_beats * 2):
                t = beat * (beat_sec / 2)
                if beat % 2 == 1:
                    t += swing
                start = int(t * sr)
                hh = hihat(sr=sr)
                hh_end = min(start + len(hh), n_total)
                if hh_end > start:
                    drum_track[start:hh_end] += params["hihat_vol"] * hh[:hh_end - start]

        for bar in range(bars):
            bar_start = bar * beats_per_bar * beat_sec
            # kicks
            for pos in drum_pat["kick_pos"]:
                t = bar_start + pos * beat_sec
                start = int(t * sr)
                k = kick(sr=sr)
                end = min(start + len(k), n_total)
                if end > start:
                    drum_track[start:end] += params["kick_vol"] * k[:end - start]
            # snares
            for pos in drum_pat["snare_pos"]:
                t = bar_start + pos * beat_sec
                start = int(t * sr)
                s = snare(sr=sr)
                end = min(start + len(s), n_total)
                if end > start:
                    drum_track[start:end] += params["snare_vol"] * s[:end - start]

    # ---- VINYL NOISE ----
    if params["enable_vinyl"]:
        noise = np.random.normal(0, 1, n_total)
        noise = np.convolve(noise, np.ones(30) / 30, mode="same")
        vinyl = params["vinyl_amount"] * noise
    else:
        vinyl = np.zeros(n_total)

    # ---- WARM PAD ----
    if params["enable_pad"]:
        t_full = np.linspace(0, total_duration, n_total, endpoint=False)
        warm_pad = params["pad_vol"] * (
            0.5 * np.sin(2 * np.pi * midi_to_freq(48) * t_full) +
            0.3 * np.sin(2 * np.pi * midi_to_freq(55) * t_full) +
            0.2 * np.sin(2 * np.pi * midi_to_freq(60) * t_full)
        )
    else:
        warm_pad = np.zeros(n_total)

    # ---- STEREO MIX ----
    mel_st = pan_stereo(melody_track, params["melody_pan"])
    chd_st = pan_stereo(chord_track, params["chord_pan"])
    bas_st = pan_stereo(bass_track, 0.0)
    drm_st = pan_stereo(drum_track, 0.05)
    fx_st = pan_stereo(vinyl + warm_pad, -0.1)

    mix_st = mel_st + chd_st + bas_st + drm_st + fx_st

    # ---- DELAY ----
    if params["enable_delay"]:
        left = add_delay(mix_st[:, 0], delay_time=params["delay_time_l"], feedback=params["delay_fb"], sr=sr)
        right = add_delay(mix_st[:, 1], delay_time=params["delay_time_r"], feedback=params["delay_fb"] * 0.88, sr=sr)
        stereo = np.column_stack([left, right])
    else:
        stereo = mix_st

    # smooth
    stereo[:, 0] = np.convolve(stereo[:, 0], np.ones(4) / 4, mode="same")
    stereo[:, 1] = np.convolve(stereo[:, 1], np.ones(4) / 4, mode="same")

    # normalize
    peak = np.max(np.abs(stereo)) + 1e-9
    stereo = stereo / peak
    return stereo, sr


def stereo_to_wav_bytes(stereo, sr):
    """Convert stereo float array to WAV bytes."""
    stereo_int16 = np.int16(stereo * 32767)
    buf = BytesIO()
    # WAV header
    n_samples = stereo_int16.shape[0]
    n_channels = 2
    sample_width = 2
    data_size = n_samples * n_channels * sample_width
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))  # PCM
    buf.write(struct.pack('<H', n_channels))
    buf.write(struct.pack('<I', sr))
    buf.write(struct.pack('<I', sr * n_channels * sample_width))
    buf.write(struct.pack('<H', n_channels * sample_width))
    buf.write(struct.pack('<H', 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    buf.write(stereo_int16.tobytes())
    buf.seek(0)
    return buf


# ======================================
# SIDEBAR - PARAMETERS
# ======================================
with st.sidebar:
    st.markdown('<p style="font-family: Outfit; font-weight: 200; font-size: 1.4rem; color: #e8e4df; letter-spacing: -0.5px;">🌙 parameters</p>', unsafe_allow_html=True)

    st.markdown("### ⏱ Tempo & Structure")
    bpm = st.slider("BPM", 55, 110, 78, 1)
    bars = st.slider("Bars", 4, 32, 8, 1)
    seed_input = st.number_input("Seed (0 = random)", min_value=0, max_value=99999, value=0, step=1)
    seed_val = None if seed_input == 0 else int(seed_input)

    st.markdown("### 🎹 Harmony")
    scale_choice = st.selectbox("Scale", list(SCALE_PRESETS.keys()), index=0)
    chord_choice = st.selectbox("Chord Progression", list(CHORD_PRESETS.keys()), index=0)

    st.markdown("### 🥁 Rhythm")
    drum_pattern = st.selectbox("Drum Pattern", list(DRUM_PATTERNS.keys()), index=0)
    swing = st.slider("Swing", 0.0, 0.15, 0.07, 0.01)

    st.markdown("### 🎚 Mixer")
    col_a, col_b = st.columns(2)
    with col_a:
        enable_melody = st.checkbox("Melody", True)
        enable_chords = st.checkbox("Chords", True)
        enable_bass = st.checkbox("Bass", True)
    with col_b:
        enable_drums = st.checkbox("Drums", True)
        enable_vinyl = st.checkbox("Vinyl Noise", True)
        enable_pad = st.checkbox("Warm Pad", True)

    st.markdown("### 🔊 Volumes")
    melody_vol = st.slider("Melody", 0.0, 0.5, 0.22, 0.01)
    chord_vol = st.slider("Chords", 0.0, 0.3, 0.12, 0.01)
    bass_vol = st.slider("Bass", 0.0, 0.5, 0.28, 0.01)
    kick_vol = st.slider("Kick", 0.0, 1.5, 0.9, 0.05)
    snare_vol = st.slider("Snare", 0.0, 1.0, 0.35, 0.05, key="snare_v")
    hihat_vol = st.slider("Hi-Hat", 0.0, 0.5, 0.10, 0.01)
    enable_hihats = st.checkbox("Enable Hi-Hats", True)

    st.markdown("### 🎨 Character")
    melody_density = st.slider("Melody Density", 0.1, 1.0, 0.75, 0.05)
    melody_range = st.slider("Melody Jump Range (semitones)", 1, 12, 5, 1)
    melody_detune = st.slider("Melody Detune (cents)", 0, 20, 8, 1)
    chord_detune = st.slider("Chord Detune (cents)", 0, 15, 4, 1)
    bass_fifth = st.slider("Bass Fifth Chance", 0.0, 0.6, 0.25, 0.05)
    vinyl_amt = st.slider("Vinyl Noise Amount", 0.0, 0.06, 0.015, 0.002)
    pad_vol = st.slider("Pad Volume", 0.0, 0.15, 0.05, 0.005)

    st.markdown("### 🌀 Effects")
    enable_delay = st.checkbox("Stereo Delay", True)
    delay_time_l = st.slider("Delay Time L (sec)", 0.1, 0.8, 0.36, 0.02)
    delay_time_r = st.slider("Delay Time R (sec)", 0.1, 0.8, 0.42, 0.02)
    delay_fb = st.slider("Delay Feedback", 0.0, 0.45, 0.16, 0.02)

    st.markdown("### 🎧 Pan")
    melody_pan = st.slider("Melody Pan", -1.0, 1.0, 0.25, 0.05)
    chord_pan = st.slider("Chord Pan", -1.0, 1.0, -0.2, 0.05)


# ======================================
# MAIN CONTENT
# ======================================
st.markdown('<p class="hero-title">lofi machine</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">rule-based beat generator with controlled randomness</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    generate_btn = st.button("✦  Generate Beat", use_container_width=True)
with col2:
    st.markdown(f'<span class="status-badge status-ready">{bpm} bpm · {bars} bars</span>', unsafe_allow_html=True)

if generate_btn:
    params = {
        "bpm": bpm,
        "bars": bars,
        "seed": seed_val,
        "scale": scale_choice,
        "chord_preset": chord_choice,
        "drum_pattern": drum_pattern,
        "swing": swing,
        "enable_melody": enable_melody,
        "enable_chords": enable_chords,
        "enable_bass": enable_bass,
        "enable_drums": enable_drums,
        "enable_vinyl": enable_vinyl,
        "enable_pad": enable_pad,
        "enable_hihats": enable_hihats,
        "melody_vol": melody_vol,
        "chord_vol": chord_vol,
        "bass_vol": bass_vol,
        "kick_vol": kick_vol,
        "snare_vol": snare_vol,
        "hihat_vol": hihat_vol,
        "melody_density": melody_density,
        "melody_range": melody_range,
        "melody_detune": melody_detune,
        "chord_detune": chord_detune,
        "bass_fifth_chance": bass_fifth,
        "vinyl_amount": vinyl_amt,
        "pad_vol": pad_vol,
        "enable_delay": enable_delay,
        "delay_time_l": delay_time_l,
        "delay_time_r": delay_time_r,
        "delay_fb": delay_fb,
        "melody_pan": melody_pan,
        "chord_pan": chord_pan,
    }

    with st.spinner("generating your beat..."):
        stereo, sr = generate_beat(params)
        wav_buf = stereo_to_wav_bytes(stereo, sr)

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    st.audio(wav_buf, format="audio/wav")

    st.download_button(
        label="⬇  Download WAV",
        data=wav_buf,
        file_name=f"lofi_{bpm}bpm_{bars}bars.wav",
        mime="audio/wav",
        use_container_width=True,
    )

    # show what was used
    with st.expander("generation details"):
        used_seed = seed_val if seed_val else "random"
        st.code(f"""seed: {used_seed}
bpm: {bpm}  |  bars: {bars}  |  scale: {scale_choice}
chords: {chord_choice}
drums: {drum_pattern}  |  swing: {swing}
delay L: {delay_time_l}s  R: {delay_time_r}s  fb: {delay_fb}""", language="yaml")

else:
    st.markdown("""
    <div style="
        margin-top: 3rem;
        padding: 3rem 2rem;
        text-align: center;
        border: 1px dashed #2a2a30;
        border-radius: 16px;
        background: #12121488;
    ">
        <p style="font-family: 'Outfit', sans-serif; font-weight: 200; font-size: 1.5rem; color: #5a5854; margin-bottom: 0.5rem;">
            adjust parameters in the sidebar
        </p>
        <p style="font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #3a3834; letter-spacing: 2px; text-transform: uppercase;">
            then hit generate
        </p>
    </div>
    """, unsafe_allow_html=True)
