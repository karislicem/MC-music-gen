import streamlit as st
import numpy as np
import random
from io import BytesIO
import struct
import math

# ═══════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════
st.set_page_config(page_title="lofi machine", page_icon="🌙", layout="centered", initial_sidebar_state="collapsed")

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
.block-container{max-width:640px!important;padding-top:4rem!important}
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
.stSelectbox label{font-family:var(--font)!important;font-weight:400!important;font-size:.85rem!important;color:var(--t2)!important}
.stSelectbox>div>div{background:var(--sf2)!important;border:1px solid var(--bd)!important;border-radius:10px!important;color:var(--t1)!important}
.stSlider label{font-family:var(--font)!important;font-weight:400!important;font-size:.85rem!important;color:var(--t2)!important}
.stSlider>div>div>div>div{background-color:var(--t1)!important}
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
# AUDIO SYNTHESIS
# ═══════════════════════════════════════
def midi_to_freq(m):
    return 440.0 * (2 ** ((m - 69) / 12))

def adsr(n, a=0.01, d=0.08, sus=0.7, r=0.12):
    ai,di,ri = int(n*a), int(n*d), int(n*r)
    si = max(0, n-ai-di-ri)
    env = np.zeros(n)
    if ai>0: env[:ai]=np.linspace(0,1,ai,endpoint=False)
    if di>0: env[ai:ai+di]=np.linspace(1,sus,di,endpoint=False)
    if si>0: env[ai+di:ai+di+si]=sus
    if ri>0: env[ai+di+si:]=np.linspace(sus,0,n-(ai+di+si),endpoint=False)
    return env

def soft_synth(freq, dur, sr=44100, detune=6):
    n=int(dur*sr); t=np.linspace(0,dur,n,endpoint=False)
    f2=freq*(2**(detune/1200))
    w=.50*np.sin(2*np.pi*freq*t)+.22*np.sin(2*np.pi*f2*t)+.18*np.sin(2*np.pi*2*freq*t)+.08*np.sin(2*np.pi*.5*freq*t)
    return np.convolve(w,np.ones(8)/8,mode="same")*adsr(n,a=.02,d=.08,sus=.65,r=.18)

def bass_synth(freq, dur, sr=44100):
    n=int(dur*sr); t=np.linspace(0,dur,n,endpoint=False)
    w=.7*np.sin(2*np.pi*freq*t)+.2*np.sin(2*np.pi*2*freq*t)
    return np.convolve(w,np.ones(12)/12,mode="same")*adsr(n,a=.005,d=.08,sus=.75,r=.1)

def kick_sound(sr=44100):
    n=int(.18*sr); t=np.linspace(0,.18,n,endpoint=False)
    return .9*np.sin(2*np.pi*(150*np.exp(-22*t)+35)*t)*np.exp(-18*t)

def snare_sound(sr=44100):
    n=int(.12*sr); t=np.linspace(0,.12,n,endpoint=False)
    s=.75*np.random.normal(0,1,n)+.25*np.sin(2*np.pi*180*t)
    return .35*np.convolve(s,np.ones(6)/6,mode="same")*np.exp(-28*t)

def hihat_sound(sr=44100):
    n=int(.05*sr); t=np.linspace(0,.05,n,endpoint=False)
    return .10*np.random.uniform(-1,1,n)*np.exp(-70*t)

def add_delay(sig, dt=.33, fb=.22, sr=44100):
    ds=int(dt*sr); out=np.copy(sig)
    for i in range(ds,len(sig)): out[i]+=fb*out[i-ds]
    return out

def pan_stereo(mono, p):
    return np.column_stack([mono*np.sqrt((1-p)/2), mono*np.sqrt((1+p)/2)])


# ═══════════════════════════════════════
# MCMC ENGINE
# ═══════════════════════════════════════

# ── Energy functions ──
# Lower energy = more musically desirable

def melody_energy(note, prev_note, chord_notes, scale_notes, temperature):
    """
    Energy of a single melody note given context.
    Terms:
      - Consonance: is it a chord tone? (strong pull)
      - Scale fit:  is it in the scale? (moderate pull)
      - Smoothness: small intervals preferred (stepwise motion)
      - Range:      penalize extreme registers
    """
    e = 0.0

    # consonance — chord tone = 0 penalty, near chord tone = small, far = large
    chord_set = set(n % 12 for n in chord_notes)
    if note % 12 in chord_set:
        e += 0.0
    else:
        # distance to nearest chord tone (pitch class)
        min_dist = min(min(abs(note%12 - c), 12-abs(note%12 - c)) for c in chord_set)
        e += min_dist * 1.5

    # scale membership
    scale_set = set(n % 12 for n in scale_notes)
    if note % 12 not in scale_set:
        e += 3.0

    # interval smoothness (prefer steps of 1-3 semitones)
    if prev_note is not None:
        interval = abs(note - prev_note)
        if interval == 0:
            e += 0.8   # repetition slight penalty
        elif interval <= 2:
            e += 0.0   # stepwise = ideal
        elif interval <= 4:
            e += 0.3   # small skip ok
        elif interval <= 7:
            e += 1.0   # larger skip
        else:
            e += interval * 0.4  # big jumps penalized

    # register — prefer 60-84 range (middle)
    if note < 55 or note > 88:
        e += 4.0
    elif note < 60 or note > 84:
        e += 1.0

    return e


def bass_energy(note, root, prev_bass, chord_notes):
    """
    Energy for bass note selection.
    Bass should strongly prefer root, fifth, or chord tones.
    """
    e = 0.0

    # root preference
    if note % 12 == root % 12:
        e += 0.0
    elif note % 12 == (root + 7) % 12:  # fifth
        e += 0.5
    elif note % 12 in set(n % 12 for n in chord_notes):
        e += 1.5
    else:
        e += 5.0

    # smooth bass motion
    if prev_bass is not None:
        interval = abs(note - prev_bass)
        if interval <= 2: e += 0.0
        elif interval <= 5: e += 0.5
        elif interval <= 7: e += 1.0
        else: e += interval * 0.3

    # bass register — prefer 36-55
    if note < 33 or note > 60:
        e += 5.0
    elif note < 36 or note > 55:
        e += 1.5

    return e


def rhythm_energy(pattern, base_pattern, density_target):
    """
    Energy for a 16-step drum pattern (kick or snare).
    pattern: list of 16 booleans
    base_pattern: the 'anchor' positions that are strongly preferred
    density_target: desired fraction of active steps (0-1)
    """
    e = 0.0

    # anchor alignment — base pattern positions should be on
    for i in base_pattern:
        if i < len(pattern) and not pattern[i]:
            e += 3.0  # missing an anchor = high cost

    # density match
    actual_density = sum(pattern) / len(pattern)
    e += abs(actual_density - density_target) * 8.0

    # syncopation bonus — hits on off-beats (odd indices) are interesting
    on_offbeat = sum(1 for i in range(len(pattern)) if pattern[i] and i % 2 == 1)
    e -= on_offbeat * 0.3  # reward (negative energy)

    # no two adjacent hits for snare (too busy)
    for i in range(len(pattern)-1):
        if pattern[i] and pattern[i+1]:
            e += 1.0

    return e


def voicing_energy(voicing, prev_voicing):
    """
    Energy for chord voicing (list of MIDI notes).
    Prefers smooth voice leading, good spread, and reasonable register.
    """
    e = 0.0

    # spread — prefer voicings spanning 10-20 semitones
    spread = max(voicing) - min(voicing)
    if spread < 7:
        e += 3.0
    elif spread < 10:
        e += 1.0
    elif spread > 24:
        e += 2.0
    # sweet spot 12-20
    elif 12 <= spread <= 20:
        e -= 0.5

    # voice leading from previous chord
    if prev_voicing is not None and len(prev_voicing) == len(voicing):
        total_motion = sum(abs(a - b) for a, b in zip(sorted(voicing), sorted(prev_voicing)))
        e += total_motion * 0.15  # prefer minimal motion

    # register — keep in reasonable piano range
    for n in voicing:
        if n < 36 or n > 84:
            e += 2.0
        elif n < 42 or n > 78:
            e += 0.5

    # no clusters — minimum distance between adjacent notes
    sv = sorted(voicing)
    for i in range(len(sv)-1):
        gap = sv[i+1] - sv[i]
        if gap < 2:
            e += 2.0  # too close (cluster)
        elif gap < 3:
            e += 0.5

    return e


# ── Metropolis-Hastings Sampler ──

def mh_sample(current_state, energy_fn, proposal_fn, temperature, n_steps=20):
    """
    Run n_steps of Metropolis-Hastings.
    Returns final state and acceptance count.

    energy_fn(state) → float
    proposal_fn(state) → new_state
    temperature: controls acceptance (low=strict, high=exploratory)
    """
    state = current_state
    current_e = energy_fn(state)
    accepted = 0

    # temperature floor to avoid division by zero
    T = max(temperature, 0.01)

    for _ in range(n_steps):
        proposal = proposal_fn(state)
        proposal_e = energy_fn(proposal)
        delta = proposal_e - current_e

        # Metropolis criterion
        if delta <= 0:
            # always accept improvements
            state = proposal
            current_e = proposal_e
            accepted += 1
        else:
            # accept worse states with probability exp(-delta/T)
            accept_prob = math.exp(-delta / T)
            if random.random() < accept_prob:
                state = proposal
                current_e = proposal_e
                accepted += 1

    return state, accepted


# ── Proposal distributions ──

def propose_melody_note(current_note):
    """Propose a nearby note (±1 to ±5 semitones)."""
    step = random.choice([-5,-4,-3,-2,-1,1,2,3,4,5])
    return current_note + step

def propose_bass_note(current_note):
    """Propose bass note — smaller range, prefer common intervals."""
    step = random.choice([-7,-5,-3,-2,-1,0,1,2,3,5,7])
    return current_note + step

def propose_rhythm_flip(pattern):
    """Flip one random step in the pattern."""
    new = list(pattern)
    i = random.randint(0, len(new)-1)
    new[i] = not new[i]
    return new

def propose_voicing(voicing):
    """Move one random voice by ±1 to ±3 semitones."""
    new = list(voicing)
    i = random.randint(0, len(new)-1)
    new[i] += random.choice([-3,-2,-1,1,2,3])
    return new


# ═══════════════════════════════════════
# MOOD PRESETS
# ═══════════════════════════════════════
MOODS = {
    "Rainy Window": {
        "description": "Melancholic chords, sparse melody, slow swing",
        "bpm_range": (62, 74),
        "scale": [57,60,62,64,67,69],
        "chords": [[45,52,57,60],[41,48,53,57],[48,52,55,60],[43,47,50,55]],
        "progression": [0,1,2,3,0,1,2,3],
        "base_kicks": [0,8],         # 16th-note indices
        "base_snares": [4,12],
        "swing": 0.08,
        "melody_density": 0.55,
        "vinyl": 0.02, "pad": 0.06,
    },
    "Sunday Morning": {
        "description": "Warm major chords, playful melody, gentle groove",
        "bpm_range": (72, 84),
        "scale": [60,62,64,67,69,72],
        "chords": [[48,55,60,64],[45,52,57,60],[41,48,52,57],[43,50,55,59]],
        "progression": [0,1,2,3,0,1,3,2],
        "base_kicks": [0,7,8],
        "base_snares": [4,12],
        "swing": 0.06,
        "melody_density": 0.75,
        "vinyl": 0.015, "pad": 0.04,
    },
    "Late Night Drive": {
        "description": "Dark jazzy voicings, deep bass, heavy swing",
        "bpm_range": (66, 78),
        "scale": [62,64,65,67,69,71,72],
        "chords": [[50,57,60,64],[43,50,55,60],[48,55,59,64],[41,48,52,57]],
        "progression": [0,1,2,3,2,0,1,3],
        "base_kicks": [0,3,10],
        "base_snares": [4,12],
        "swing": 0.10,
        "melody_density": 0.60,
        "vinyl": 0.018, "pad": 0.05,
    },
    "Café in Kyoto": {
        "description": "Pentatonic melody, minimal drums, lots of space",
        "bpm_range": (68, 80),
        "scale": [64,67,69,71,74,76],
        "chords": [[40,47,52,55],[48,55,60,64],[45,52,57,60],[47,54,59,62]],
        "progression": [0,1,2,3,0,2,1,3],
        "base_kicks": [0,10],
        "base_snares": [8],
        "swing": 0.05,
        "melody_density": 0.50,
        "vinyl": 0.022, "pad": 0.07,
    },
    "Rooftop Sunset": {
        "description": "Bright and uplifting, open chords, airy melody",
        "bpm_range": (76, 88),
        "scale": [65,67,69,71,72,74,76],
        "chords": [[41,48,52,57],[40,47,52,55],[50,57,60,65],[48,55,60,64]],
        "progression": [0,1,2,3,0,1,3,2],
        "base_kicks": [0,7,8,11],
        "base_snares": [4,12,14],
        "swing": 0.04,
        "melody_density": 0.80,
        "vinyl": 0.012, "pad": 0.03,
    },
}

# ═══════════════════════════════════════
# FULL MCMC GENERATOR
# ═══════════════════════════════════════
def generate_beat(mood_name, bpm, bars, density, temperature):
    sr = 44100
    mood = MOODS[mood_name]
    beat_sec = 60 / bpm
    bpb = 4
    total_beats = bars * bpb
    total_dur = total_beats * beat_sec
    n_total = int(total_dur * sr)

    melody_track = np.zeros(n_total)
    chord_track = np.zeros(n_total)
    bass_track = np.zeros(n_total)
    drum_track = np.zeros(n_total)

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

    # MCMC temperature scaling
    # User temperature 0.0-1.0 maps to internal T for each chain
    T_melody = 0.3 + temperature * 3.0     # 0.3 (strict) to 3.3 (wild)
    T_bass = 0.2 + temperature * 2.0       # bass stays tighter
    T_rhythm = 0.3 + temperature * 2.5
    T_voicing = 0.2 + temperature * 2.0

    # MH burn-in steps and sampling steps
    burn_in = 15
    sample_steps = 20

    # Stats tracking
    stats = {"melody_accepted": 0, "melody_total": 0,
             "bass_accepted": 0, "bass_total": 0,
             "rhythm_accepted": 0, "rhythm_total": 0,
             "voicing_accepted": 0, "voicing_total": 0}

    # ─── MCMC: CHORD VOICINGS ───
    # For each bar, sample a voicing via MH
    prev_voicing = None
    bar_voicings = []
    for bar in range(bars):
        base_chord = chords_base[prog[bar % len(prog)]]

        def voicing_e(v):
            return voicing_energy(v, prev_voicing)

        # initialize with base chord
        state = list(base_chord)

        # burn-in
        mh_sample(state, voicing_e, propose_voicing, T_voicing, burn_in)
        # sample
        state, acc = mh_sample(state, voicing_e, propose_voicing, T_voicing, sample_steps)
        stats["voicing_accepted"] += acc
        stats["voicing_total"] += sample_steps

        # ensure voicing preserves chord identity (same pitch classes)
        # by snapping to nearest note with correct pitch class
        base_pcs = [n % 12 for n in base_chord]
        final_voicing = []
        for i, pc in enumerate(base_pcs):
            # find nearest note to state[i] with pitch class pc
            target = state[i]
            best = target
            best_dist = 999
            for offset in range(-12, 13):
                candidate = target + offset
                if candidate % 12 == pc and abs(offset) < best_dist:
                    best = candidate
                    best_dist = abs(offset)
            final_voicing.append(best)

        bar_voicings.append(final_voicing)
        prev_voicing = final_voicing

    # ─── RENDER CHORDS ───
    for bar in range(bars):
        voicing = bar_voicings[bar]
        st_t = bar * bpb * beat_sec
        dur = bpb * beat_sec
        for note in voicing:
            wave = soft_synth(midi_to_freq(note), dur, sr, detune=4)
            s = int(st_t * sr); e = s + len(wave)
            if e <= n_total:
                chord_track[s:e] += chd_vol * wave

    # ─── MCMC: MELODY ───
    positions = [0.0, 0.75, 1.5, 2.5, 3.25]
    prev_melody_note = random.choice(scale) + 12  # start in upper octave

    for bar in range(bars):
        voicing = bar_voicings[bar]
        base_chord = chords_base[prog[bar % len(prog)]]

        for pos in positions:
            # density gate — but MCMC-influenced
            # at high temperature, occasionally add extra notes too
            skip_threshold = 1.0 - mel_dens
            if random.random() < skip_threshold * (1.0 - temperature * 0.3):
                continue

            # MCMC sample for this note
            current = prev_melody_note

            def mel_e(note):
                return melody_energy(note, prev_melody_note, voicing, scale, T_melody)

            # burn-in then sample
            mh_sample(current, mel_e, propose_melody_note, T_melody, burn_in)
            note, acc = mh_sample(current, mel_e, propose_melody_note, T_melody, sample_steps)
            stats["melody_accepted"] += acc
            stats["melody_total"] += sample_steps

            # duration also sampled with slight temperature influence
            dur_choices = [0.4, 0.5, 0.75, 1.0]
            if temperature > 0.6:
                dur_choices += [0.25, 1.5]  # more variety at high T
            dur = random.choice(dur_choices) * beat_sec

            s = int((bar * bpb + pos) * beat_sec * sr)
            wave = soft_synth(midi_to_freq(note), dur, sr, detune=8)
            e = min(s + len(wave), n_total)
            if e > s:
                melody_track[s:e] += mel_vol * wave[:e-s]

            prev_melody_note = note

    # ─── MCMC: BASS ───
    prev_bass_note = None
    bass_positions = [0, 1.5, 2.0, 3.0]
    bass_durations = [1.0, 0.4, 0.8, 0.7]

    for bar in range(bars):
        base_chord = chords_base[prog[bar % len(prog)]]
        root = base_chord[0]

        for pos, ln in zip(bass_positions, bass_durations):
            current = root if prev_bass_note is None else prev_bass_note

            def bass_e(note):
                return bass_energy(note, root, prev_bass_note, base_chord)

            mh_sample(current, bass_e, propose_bass_note, T_bass, burn_in)
            note, acc = mh_sample(current, bass_e, propose_bass_note, T_bass, sample_steps)
            stats["bass_accepted"] += acc
            stats["bass_total"] += sample_steps

            s = int((bar * bpb + pos) * beat_sec * sr)
            wave = bass_synth(midi_to_freq(note), ln * beat_sec, sr)
            e = min(s + len(wave), n_total)
            if e > s:
                bass_track[s:e] += bas_vol * wave[:e-s]

            prev_bass_note = note

    # ─── MCMC: DRUM PATTERNS ───
    # Sample kick and snare patterns per bar using MH
    kick_density_target = len(mood["base_kicks"]) / 16 + density * 0.08
    snare_density_target = len(mood["base_snares"]) / 16 + density * 0.04

    for bar in range(bars):
        bar_time = bar * bpb * beat_sec
        sixteenth = beat_sec / 4

        # ── Kick pattern via MCMC ──
        init_kick = [False] * 16
        for i in mood["base_kicks"]:
            if i < 16: init_kick[i] = True

        def kick_e(pat):
            return rhythm_energy(pat, mood["base_kicks"], kick_density_target)

        mh_sample(init_kick, kick_e, propose_rhythm_flip, T_rhythm, burn_in)
        kick_pattern, acc = mh_sample(init_kick, kick_e, propose_rhythm_flip, T_rhythm, sample_steps)
        stats["rhythm_accepted"] += acc
        stats["rhythm_total"] += sample_steps

        # ── Snare pattern via MCMC ──
        init_snare = [False] * 16
        for i in mood["base_snares"]:
            if i < 16: init_snare[i] = True

        def snare_e(pat):
            return rhythm_energy(pat, mood["base_snares"], snare_density_target)

        mh_sample(init_snare, snare_e, propose_rhythm_flip, T_rhythm, burn_in)
        snare_pattern, acc = mh_sample(init_snare, snare_e, propose_rhythm_flip, T_rhythm, sample_steps)
        stats["rhythm_accepted"] += acc
        stats["rhythm_total"] += sample_steps

        # ── Render drum bar ──
        for step in range(16):
            t = bar_time + step * sixteenth
            # swing on off-beats
            if step % 2 == 1:
                t += swing * beat_sec

            s = int(t * sr)

            # hi-hat on every step
            hh = hihat_sound(sr=sr)
            e = min(s + len(hh), n_total)
            if e > s: drum_track[s:e] += hh[:e-s]

            # kick
            if kick_pattern[step]:
                k = kick_sound(sr=sr)
                e = min(s + len(k), n_total)
                if e > s: drum_track[s:e] += 0.9 * k[:e-s]

            # snare
            if snare_pattern[step]:
                sn = snare_sound(sr=sr)
                e = min(s + len(sn), n_total)
                if e > s: drum_track[s:e] += sn[:e-s]

    # ─── TEXTURE ───
    noise = np.convolve(np.random.normal(0,1,n_total), np.ones(30)/30, mode="same")
    vinyl = vinyl_amt * noise
    t_f = np.linspace(0, total_dur, n_total, endpoint=False)
    pad = pad_vol * (
        .5*np.sin(2*np.pi*midi_to_freq(48)*t_f) +
        .3*np.sin(2*np.pi*midi_to_freq(55)*t_f) +
        .2*np.sin(2*np.pi*midi_to_freq(60)*t_f)
    )

    # ─── STEREO MIX ───
    mix = (
        pan_stereo(melody_track, 0.25) + pan_stereo(chord_track, -0.2) +
        pan_stereo(bass_track, 0.0) + pan_stereo(drum_track, 0.05) +
        pan_stereo(vinyl + pad, -0.1)
    )
    left = add_delay(mix[:,0], .36, .16, sr)
    right = add_delay(mix[:,1], .42, .14, sr)
    stereo = np.column_stack([left, right])
    stereo[:,0] = np.convolve(stereo[:,0], np.ones(4)/4, mode="same")
    stereo[:,1] = np.convolve(stereo[:,1], np.ones(4)/4, mode="same")
    stereo /= np.max(np.abs(stereo)) + 1e-9

    return stereo, sr, stats


def to_wav(stereo, sr):
    data=np.int16(stereo*32767); buf=BytesIO()
    ns,nc,sw=data.shape[0],2,2; ds=ns*nc*sw
    buf.write(b'RIFF'); buf.write(struct.pack('<I',36+ds))
    buf.write(b'WAVE'); buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH',16,1,nc,sr,sr*nc*sw,nc*sw,16))
    buf.write(b'data'); buf.write(struct.pack('<I',ds))
    buf.write(data.tobytes()); buf.seek(0); return buf


# ═══════════════════════════════════════
# UI
# ═══════════════════════════════════════
st.markdown('<p class="site-title">lofi machine</p>', unsafe_allow_html=True)
st.markdown('<p class="site-sub">Markov Chain Monte Carlo meets lo-fi hip hop.</p>', unsafe_allow_html=True)
st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ── Mood ──
st.markdown('<p class="section-label">Mood</p>', unsafe_allow_html=True)
st.markdown('<p class="hint">The seed of everything — harmonic palette, scale, rhythmic skeleton, and atmosphere. MCMC chains explore variations around this foundation.</p>', unsafe_allow_html=True)
mood_names = list(MOODS.keys())
mood = st.selectbox("Mood", mood_names, index=1,
    format_func=lambda x: f"{x}  —  {MOODS[x]['description']}",
    label_visibility="collapsed")

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

# ── Tempo ──
lo, hi = MOODS[mood]["bpm_range"]
st.markdown('<p class="section-label">Tempo</p>', unsafe_allow_html=True)
st.markdown(f'<p class="hint">Sweet spot for this mood: {lo}–{hi} BPM. Go outside if you want — the chains will adapt.</p>', unsafe_allow_html=True)
bpm = st.slider("BPM", 55, 110, (lo+hi)//2, 1, label_visibility="collapsed")

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

# ── Duration ──
st.markdown('<p class="section-label">Duration</p>', unsafe_allow_html=True)
st.markdown('<p class="hint">How long the piece runs. More bars = more room for the Markov chains to evolve and find interesting patterns.</p>', unsafe_allow_html=True)
bars = st.select_slider("Bars", options=[4,8,12,16,24,32], value=8, label_visibility="collapsed")

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

# ── Density ──
st.markdown('<p class="section-label">Density</p>', unsafe_allow_html=True)
st.markdown('<p class="hint">How full the arrangement is. Affects note frequency, layer volumes, and how much texture fills the space.</p>', unsafe_allow_html=True)
density = st.slider("Density", 0.0, 1.0, 0.6, 0.05, label_visibility="collapsed")

st.markdown('<hr class="sep-light">', unsafe_allow_html=True)

# ── Temperature ──
st.markdown('<p class="section-label">Temperature</p>', unsafe_allow_html=True)
st.markdown('<p class="hint">The soul of the MCMC sampler. Low = the chains converge to safe, consonant choices. High = they explore dissonance, wider intervals, and unexpected rhythms. This is the Metropolis-Hastings acceptance threshold — it literally controls how likely the algorithm is to accept a "worse" musical choice.</p>', unsafe_allow_html=True)
temperature = st.slider("Temperature", 0.0, 1.0, 0.35, 0.05, label_visibility="collapsed")

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
    temperature = round(random.uniform(0.1, 0.8), 2)

if generate_btn or random_btn:
    random.seed()
    np.random.seed()

    with st.spinner(""):
        stereo, sr, stats = generate_beat(mood, bpm, bars, density, temperature)
        wav = to_wav(stereo, sr)

    dur_sec = round(bars * 4 * (60 / bpm), 1)

    st.markdown('<hr class="sep">', unsafe_allow_html=True)
    st.audio(wav, format="audio/wav")
    st.markdown(
        f'<p class="gen-info">{mood} · {bpm} bpm · {bars} bars · {dur_sec}s · T={temperature}</p>',
        unsafe_allow_html=True,
    )

    # MCMC diagnostics
    def ar(a, t):
        return f"{a/t*100:.0f}%" if t > 0 else "—"

    st.markdown(f"""<div class="mcmc-stats">
<strong style="color:#6e6e73">MCMC Diagnostics</strong><br>
Melody chain &nbsp;→&nbsp; {stats["melody_accepted"]}/{stats["melody_total"]} accepted ({ar(stats["melody_accepted"], stats["melody_total"])})<br>
Bass chain &nbsp;&nbsp;&nbsp;→&nbsp; {stats["bass_accepted"]}/{stats["bass_total"]} accepted ({ar(stats["bass_accepted"], stats["bass_total"])})<br>
Rhythm chain &nbsp;→&nbsp; {stats["rhythm_accepted"]}/{stats["rhythm_total"]} accepted ({ar(stats["rhythm_accepted"], stats["rhythm_total"])})<br>
Voicing chain →&nbsp; {stats["voicing_accepted"]}/{stats["voicing_total"]} accepted ({ar(stats["voicing_accepted"], stats["voicing_total"])})
</div>""", unsafe_allow_html=True)

    st.markdown("", unsafe_allow_html=True)
    st.download_button(
        "Download WAV", wav,
        file_name=f"lofi_mcmc_{mood.lower().replace(' ','_')}_{bpm}bpm_T{temperature}.wav",
        mime="audio/wav", use_container_width=True,
    )
else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🎧</div>
        <p class="empty-text">Four Markov chains, waiting to compose.</p>
        <p class="empty-hint">Or let randomness decide everything.</p>
    </div>
    """, unsafe_allow_html=True)
