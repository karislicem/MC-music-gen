import streamlit as st
import numpy as np
import random
from io import BytesIO
import struct

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="lofi machine",
    page_icon="🌙",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────
# APPLE-GRADE CSS
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500&display=swap');

:root {
    --bg: #000000;
    --surface: #0a0a0a;
    --surface-2: #111111;
    --border: #222222;
    --border-light: #2a2a2a;
    --text-1: #f5f5f7;
    --text-2: #a1a1a6;
    --text-3: #6e6e73;
    --text-4: #48484a;
    --accent-dim: rgba(255,255,255,0.06);
    --radius-sm: 10px;
    --radius: 14px;
    --radius-lg: 20px;
    --font: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --mono: SFMono-Regular, Menlo, monospace;
}

.stApp {
    background: var(--bg) !important;
    font-family: var(--font) !important;
}

header[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 640px !important; padding-top: 4rem !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Typography ── */
.site-title {
    font-family: var(--font);
    font-weight: 200;
    font-size: 2.8rem;
    color: var(--text-1);
    letter-spacing: -1.5px;
    line-height: 1;
    margin: 0;
    text-align: center;
}

.site-sub {
    font-family: var(--font);
    font-weight: 300;
    font-size: 1.05rem;
    color: var(--text-3);
    text-align: center;
    margin-top: 8px;
    letter-spacing: -0.2px;
}

.sep { border: none; border-top: 1px solid var(--border); margin: 2.4rem 0; }
.sep-light { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }

.section-label {
    font-family: var(--font);
    font-weight: 500;
    font-size: 0.7rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 16px;
}

.hint {
    font-family: var(--font);
    font-weight: 300;
    font-size: 0.78rem;
    color: var(--text-4);
    line-height: 1.5;
    margin-top: -4px;
    margin-bottom: 18px;
}

/* ── Buttons ── */
.stButton > button {
    font-family: var(--font) !important;
    font-weight: 400 !important;
    font-size: 0.95rem !important;
    letter-spacing: -0.2px !important;
    border-radius: var(--radius) !important;
    padding: 0.7rem 1.6rem !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    border: none !important;
}

div[data-testid="stColumns"] > div:first-child .stButton > button {
    background: var(--text-1) !important;
    color: var(--bg) !important;
}
div[data-testid="stColumns"] > div:first-child .stButton > button:hover {
    background: #e0e0e0 !important;
}

div[data-testid="stColumns"] > div:nth-child(2) .stButton > button {
    background: transparent !important;
    color: var(--text-2) !important;
    border: 1px solid var(--border-light) !important;
}
div[data-testid="stColumns"] > div:nth-child(2) .stButton > button:hover {
    background: var(--accent-dim) !important;
    color: var(--text-1) !important;
    border-color: var(--text-4) !important;
}

/* Download */
.stDownloadButton > button {
    background: transparent !important;
    color: var(--text-2) !important;
    border: 1px solid var(--border-light) !important;
    font-family: var(--font) !important;
    font-weight: 400 !important;
    font-size: 0.85rem !important;
    border-radius: var(--radius) !important;
}
.stDownloadButton > button:hover {
    background: var(--accent-dim) !important;
    color: var(--text-1) !important;
}

/* ── Selectbox ── */
.stSelectbox label {
    font-family: var(--font) !important;
    font-weight: 400 !important;
    font-size: 0.85rem !important;
    color: var(--text-2) !important;
}
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-1) !important;
}

/* ── Slider ── */
.stSlider label {
    font-family: var(--font) !important;
    font-weight: 400 !important;
    font-size: 0.85rem !important;
    color: var(--text-2) !important;
}
.stSlider > div > div > div > div {
    background-color: var(--text-1) !important;
}

/* ── Audio ── */
.stAudio > div { border-radius: var(--radius) !important; }

/* ── Info badge ── */
.gen-info {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-4);
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px 10px;
    margin-top: 12px;
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 5rem 2rem; margin-top: 1rem; }
.empty-icon { font-size: 2.4rem; margin-bottom: 1rem; opacity: 0.3; }
.empty-text { font-family: var(--font); font-weight: 300; font-size: 1.1rem; color: var(--text-4); letter-spacing: -0.3px; }
.empty-hint { font-family: var(--font); font-weight: 300; font-size: 0.82rem; color: var(--text-4); opacity: 0.5; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────
# AUDIO ENGINE
# ─────────────────────────────────────
def midi_to_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12))

def adsr(n, attack=0.01, decay=0.08, sustain=0.7, release=0.12):
    a, d, r = int(n * attack), int(n * decay), int(n * release)
    s = max(0, n - a - d - r)
    env = np.zeros(n)
    if a > 0: env[:a] = np.linspace(0, 1, a, endpoint=False)
    if d > 0: env[a:a+d] = np.linspace(1, sustain, d, endpoint=False)
    if s > 0: env[a+d:a+d+s] = sustain
    if r > 0: env[a+d+s:] = np.linspace(sustain, 0, n - (a+d+s), endpoint=False)
    return env

def soft_synth(freq, dur, sr=44100, detune_cents=6):
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False)
    f2 = freq * (2 ** (detune_cents / 1200))
    wave = 0.50*np.sin(2*np.pi*freq*t) + 0.22*np.sin(2*np.pi*f2*t) + 0.18*np.sin(2*np.pi*2*freq*t) + 0.08*np.sin(2*np.pi*0.5*freq*t)
    wave = np.convolve(wave, np.ones(8)/8, mode="same")
    return wave * adsr(n, attack=0.02, decay=0.08, sustain=0.65, release=0.18)

def bass_synth(freq, dur, sr=44100):
    n = int(dur * sr)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = 0.7*np.sin(2*np.pi*freq*t) + 0.2*np.sin(2*np.pi*2*freq*t)
    wave = np.convolve(wave, np.ones(12)/12, mode="same")
    return wave * adsr(n, attack=0.005, decay=0.08, sustain=0.75, release=0.1)

def kick(sr=44100, dur=0.18):
    n = int(dur * sr); t = np.linspace(0, dur, n, endpoint=False)
    return 0.9 * np.sin(2*np.pi*(150*np.exp(-22*t)+35)*t) * np.exp(-18*t)

def snare(sr=44100, dur=0.12):
    n = int(dur * sr); t = np.linspace(0, dur, n, endpoint=False)
    sig = 0.75*np.random.normal(0,1,n) + 0.25*np.sin(2*np.pi*180*t)
    sig = np.convolve(sig, np.ones(6)/6, mode="same")
    return 0.35 * sig * np.exp(-28*t)

def hihat(sr=44100, dur=0.05):
    n = int(dur * sr); t = np.linspace(0, dur, n, endpoint=False)
    return 0.10 * np.random.uniform(-1,1,n) * np.exp(-70*t)

def add_delay(signal, delay_time=0.33, feedback=0.22, sr=44100):
    ds = int(delay_time * sr); out = np.copy(signal)
    for i in range(ds, len(signal)): out[i] += feedback * out[i - ds]
    return out

def pan_stereo(mono, pan):
    return np.column_stack([mono*np.sqrt((1-pan)/2), mono*np.sqrt((1+pan)/2)])


# ─────────────────────────────────────
# MOOD PRESETS
# ─────────────────────────────────────
MOODS = {
    "Rainy Window": {
        "description": "Melancholic chords, sparse melody, slow swing",
        "bpm_range": (62, 74),
        "scale": [57, 60, 62, 64, 67, 69],
        "chords": [[45,52,57,60],[41,48,53,57],[48,52,55,60],[43,47,50,55]],
        "progression": [0,1,2,3,0,1,2,3],
        "drum_kicks": [0.0, 2.0],
        "drum_snares": [1.0, 3.0],
        "swing": 0.08,
        "melody_density": 0.55,
        "vinyl": 0.02,
        "pad": 0.06,
    },
    "Sunday Morning": {
        "description": "Warm major chords, playful melody, gentle groove",
        "bpm_range": (72, 84),
        "scale": [60, 62, 64, 67, 69, 72],
        "chords": [[48,55,60,64],[45,52,57,60],[41,48,52,57],[43,50,55,59]],
        "progression": [0,1,2,3,0,1,3,2],
        "drum_kicks": [0.0, 1.75, 2.0],
        "drum_snares": [1.0, 3.0],
        "swing": 0.06,
        "melody_density": 0.75,
        "vinyl": 0.015,
        "pad": 0.04,
    },
    "Late Night Drive": {
        "description": "Dark jazzy voicings, deep bass, heavy swing",
        "bpm_range": (66, 78),
        "scale": [62, 64, 65, 67, 69, 71, 72],
        "chords": [[50,57,60,64],[43,50,55,60],[48,55,59,64],[41,48,52,57]],
        "progression": [0,1,2,3,2,0,1,3],
        "drum_kicks": [0.0, 0.75, 2.5],
        "drum_snares": [1.0, 3.0],
        "swing": 0.10,
        "melody_density": 0.60,
        "vinyl": 0.018,
        "pad": 0.05,
    },
    "Café in Kyoto": {
        "description": "Pentatonic melody, minimal drums, lots of space",
        "bpm_range": (68, 80),
        "scale": [64, 67, 69, 71, 74, 76],
        "chords": [[40,47,52,55],[48,55,60,64],[45,52,57,60],[47,54,59,62]],
        "progression": [0,1,2,3,0,2,1,3],
        "drum_kicks": [0.0, 2.5],
        "drum_snares": [2.0],
        "swing": 0.05,
        "melody_density": 0.50,
        "vinyl": 0.022,
        "pad": 0.07,
    },
    "Rooftop Sunset": {
        "description": "Bright and uplifting, open chords, airy melody",
        "bpm_range": (76, 88),
        "scale": [65, 67, 69, 71, 72, 74, 76],
        "chords": [[41,48,52,57],[40,47,52,55],[50,57,60,65],[48,55,60,64]],
        "progression": [0,1,2,3,0,1,3,2],
        "drum_kicks": [0.0, 1.75, 2.0, 2.75],
        "drum_snares": [1.0, 3.0, 3.5],
        "swing": 0.04,
        "melody_density": 0.80,
        "vinyl": 0.012,
        "pad": 0.03,
    },
}


# ─────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────
def generate_beat(mood_name, bpm, bars, density):
    sr = 44100
    mood = MOODS[mood_name]
    beat_sec = 60 / bpm
    bpb = 4
    total_beats = bars * bpb
    total_dur = total_beats * beat_sec
    n_total = int(total_dur * sr)

    melody_t = np.zeros(n_total)
    chord_t = np.zeros(n_total)
    bass_t = np.zeros(n_total)
    drum_t = np.zeros(n_total)

    chords = mood["chords"]
    prog = mood["progression"]
    scale = mood["scale"]
    swing = mood["swing"]

    mel_vol = 0.10 + 0.18 * density
    chd_vol = 0.06 + 0.10 * density
    bas_vol = 0.18 + 0.14 * density
    mel_dens = mood["melody_density"] * (0.5 + 0.5 * density)
    vinyl_amt = mood["vinyl"] * (0.6 + 0.4 * density)
    pad_vol = mood["pad"] * (0.4 + 0.6 * density)

    # Chords
    for bar in range(bars):
        chord = chords[prog[bar % len(prog)]]
        st_t = bar * bpb * beat_sec
        dur = bpb * beat_sec
        for note in chord:
            wave = soft_synth(midi_to_freq(note), dur, sr, detune_cents=4)
            s = int(st_t * sr); e = s + len(wave)
            if e <= n_total: chord_t[s:e] += chd_vol * wave

    # Melody
    positions = [0.0, 0.75, 1.5, 2.5, 3.25]
    for bar in range(bars):
        chord = chords[prog[bar % len(prog)]]
        chord_upper = [n + 12 for n in chord[1:]]
        last = random.choice(chord_upper)
        for pos in positions:
            if random.random() > mel_dens: continue
            pool = chord_upper + scale
            cands = [n for n in pool if abs(n - last) <= 5] or pool
            pitch = random.choice(cands)
            dur = random.choice([0.4, 0.5, 0.75, 1.0]) * beat_sec
            s = int((bar * bpb + pos) * beat_sec * sr)
            wave = soft_synth(midi_to_freq(pitch), dur, sr, detune_cents=8)
            e = min(s + len(wave), n_total)
            if e > s: melody_t[s:e] += mel_vol * wave[:e-s]
            last = pitch

    # Bass
    for bar in range(bars):
        root = chords[prog[bar % len(prog)]][0]
        for pos, ln in zip([0, 1.5, 2.0, 3.0], [1.0, 0.4, 0.8, 0.7]):
            s = int((bar * bpb + pos) * beat_sec * sr)
            p = root + 7 if random.random() < 0.25 else root
            wave = bass_synth(midi_to_freq(p), ln * beat_sec, sr)
            e = min(s + len(wave), n_total)
            if e > s: bass_t[s:e] += bas_vol * wave[:e-s]

    # Drums
    sw = swing * beat_sec
    for beat in range(total_beats * 2):
        t = beat * (beat_sec / 2)
        if beat % 2 == 1: t += sw
        s = int(t * sr)
        hh = hihat(sr=sr); e = min(s + len(hh), n_total)
        if e > s: drum_t[s:e] += hh[:e-s]

    for bar in range(bars):
        bs = bar * bpb * beat_sec
        for pos in mood["drum_kicks"]:
            s = int((bs + pos * beat_sec) * sr)
            k = kick(sr=sr); e = min(s + len(k), n_total)
            if e > s: drum_t[s:e] += 0.9 * k[:e-s]
        for pos in mood["drum_snares"]:
            s = int((bs + pos * beat_sec) * sr)
            sn = snare(sr=sr); e = min(s + len(sn), n_total)
            if e > s: drum_t[s:e] += sn[:e-s]

    # Texture
    noise = np.convolve(np.random.normal(0, 1, n_total), np.ones(30)/30, mode="same")
    vinyl = vinyl_amt * noise
    t_f = np.linspace(0, total_dur, n_total, endpoint=False)
    pad = pad_vol * (
        0.5*np.sin(2*np.pi*midi_to_freq(48)*t_f) +
        0.3*np.sin(2*np.pi*midi_to_freq(55)*t_f) +
        0.2*np.sin(2*np.pi*midi_to_freq(60)*t_f)
    )

    # Stereo mix
    mix = (
        pan_stereo(melody_t, 0.25) + pan_stereo(chord_t, -0.2) +
        pan_stereo(bass_t, 0.0) + pan_stereo(drum_t, 0.05) +
        pan_stereo(vinyl + pad, -0.1)
    )
    left = add_delay(mix[:,0], 0.36, 0.16, sr)
    right = add_delay(mix[:,1], 0.42, 0.14, sr)
    stereo = np.column_stack([left, right])
    stereo[:,0] = np.convolve(stereo[:,0], np.ones(4)/4, mode="same")
    stereo[:,1] = np.convolve(stereo[:,1], np.ones(4)/4, mode="same")
    stereo /= np.max(np.abs(stereo)) + 1e-9
    return stereo, sr

def to_wav(stereo, sr):
    data = np.int16(stereo * 32767); buf = BytesIO()
    ns, nc, sw = data.shape[0], 2, 2; ds = ns * nc * sw
    buf.write(b'RIFF'); buf.write(struct.pack('<I', 36+ds))
    buf.write(b'WAVE'); buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, nc, sr, sr*nc*sw, nc*sw, 16))
    buf.write(b'data'); buf.write(struct.pack('<I', ds))
    buf.write(data.tobytes()); buf.seek(0); return buf


# ─────────────────────────────────────
# UI
# ─────────────────────────────────────

st.markdown('<p class="site-title">lofi machine</p>', unsafe_allow_html=True)
st.markdown('<p class="site-sub">Pick a mood. Adjust to taste. Let the algorithm compose.</p>', unsafe_allow_html=True)
st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ── Mood ──
st.markdown('<p class="section-label">Mood</p>', unsafe_allow_html=True)
st.markdown('<p class="hint">Sets the harmonic palette, scale, drum pattern, and overall character. Everything else follows from here.</p>', unsafe_allow_html=True)

mood_names = list(MOODS.keys())
mood = st.selectbox(
    "Mood", mood_names, index=1,
    format_func=lambda x: f"{x}  —  {MOODS[x]['description']}",
    label_visibility="collapsed",
)

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

# ── Tempo ──
lo, hi = MOODS[mood]["bpm_range"]
st.markdown('<p class="section-label">Tempo</p>', unsafe_allow_html=True)
st.markdown(f'<p class="hint">How fast it breathes. This mood\'s sweet spot is {lo}–{hi} BPM.</p>', unsafe_allow_html=True)
bpm = st.slider("BPM", 55, 110, (lo + hi) // 2, 1, label_visibility="collapsed")

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

# ── Duration ──
st.markdown('<p class="section-label">Duration</p>', unsafe_allow_html=True)
st.markdown('<p class="hint">8 bars ≈ a short loop. 16 ≈ a full idea. 32 ≈ a long session piece.</p>', unsafe_allow_html=True)
bars = st.select_slider("Bars", options=[4, 8, 12, 16, 24, 32], value=8, label_visibility="collapsed")

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

# ── Density ──
st.markdown('<p class="section-label">Density</p>', unsafe_allow_html=True)
st.markdown('<p class="hint">From barely there to fully arranged. Controls how many notes, how loud each layer, and how much texture.</p>', unsafe_allow_html=True)
density = st.slider("Density", 0.0, 1.0, 0.6, 0.05, label_visibility="collapsed")

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ── Actions ──
col1, col2 = st.columns(2, gap="medium")
with col1:
    generate_btn = st.button("Generate", use_container_width=True)
with col2:
    random_btn = st.button("Trust the Randomness", use_container_width=True)

# ── Run ──
if random_btn:
    mood = random.choice(mood_names)
    lo, hi = MOODS[mood]["bpm_range"]
    bpm = random.randint(lo, hi)
    bars = random.choice([8, 12, 16])
    density = round(random.uniform(0.3, 0.9), 2)

if generate_btn or random_btn:
    random.seed()
    np.random.seed()

    with st.spinner(""):
        stereo, sr = generate_beat(mood, bpm, bars, density)
        wav = to_wav(stereo, sr)

    dur_sec = round(bars * 4 * (60 / bpm), 1)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.audio(wav, format="audio/wav")
    st.markdown(
        f'<p class="gen-info">{mood} · {bpm} bpm · {bars} bars · {dur_sec}s · density {density}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("", unsafe_allow_html=True)
    st.download_button(
        "Download WAV", wav,
        file_name=f"lofi_{mood.lower().replace(' ', '_')}_{bpm}bpm.wav",
        mime="audio/wav", use_container_width=True,
    )
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎧</div>
        <p class="empty-text">Your beat is one click away.</p>
        <p class="empty-hint">Or let randomness decide.</p>
    </div>
    """, unsafe_allow_html=True)
