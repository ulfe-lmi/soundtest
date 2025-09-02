#!/usr/bin/env python3
# midi_synth32.py — 32‑voice MIDI renderer with PyAudio playback, WAV export, and realtime mode.
# Quiet: on success prints ONLY the audio device list (stdout). Progress/errors go to stderr.
#
# Usage examples:
#   python midi_synth32.py --test-tone
#   python midi_synth32.py --midi song.mid --progress
#   python midi_synth32.py --midi song.mid --preview 25 --progress --save
#   python midi_synth32.py --midi song.mid --realtime --prebuffer 4 --progress
#
# Notes:
# - No copyrighted content is embedded; render your own legally obtained MIDI files.
# - Supports: tempo changes, Program Change, CC7 (volume), CC10 (pan), CC64 (sustain),
#             CC120 (All Sound Off), CC123 (All Notes Off), CC121 (Reset All Controllers).
# - Channel 10 (index 9) is treated as GM drums.

import argparse
import os
import sys
import time
import threading
import tempfile
from collections import deque
import numpy as np
import pyaudio
import wave

# ---- MIDI import ----
try:
    import mido
except Exception as e:
    sys.stderr.write(
        "This script requires the 'mido' package for MIDI parsing.\n"
        "Install it with: pip install mido\n"
        f"Import error: {e}\n"
    )
    sys.exit(1)

SR = 44100

# -------------------- silence C-level stderr safely (no ctypes) --------------------
class capture_stderr:
    """Temporarily capture fd(2) so ALSA/PortAudio noise is hidden unless error occurs."""
    def __enter__(self):
        self._old = os.dup(2)
        self._tmp = tempfile.TemporaryFile()
        os.dup2(self._tmp.fileno(), 2)
        return self
    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._old, 2)
        os.close(self._old)
        self._tmp.seek(0)
        self.data = self._tmp.read().decode(errors="ignore")
        self._tmp.close()

# ------------------------------ DSP primitives --------------------------------
def midi_to_freq(m): return 440.0 * (2.0 ** ((m - 69) / 12.0))

def one_pole_lp(x, cutoff_hz_arr, sr):
    y = np.zeros_like(x, dtype=np.float32); z = 0.0; twopi = 2.0 * np.pi
    for i in range(len(x)):
        fc = float(cutoff_hz_arr[i])
        fc = 20.0 if fc < 20.0 else (sr/2 - 200.0 if fc > (sr/2 - 200.0) else fc)
        alpha = (twopi * fc) / (twopi * fc + sr)
        z += alpha * (x[i] - z); y[i] = z
    return y

def one_pole_hp(x, cutoff_hz, sr):
    c = np.full_like(x, float(cutoff_hz), dtype=np.float32)
    return x - one_pole_lp(x, c, sr)

def soft_clip_tanh(x, drive=2.5):
    return np.tanh(x * drive).astype(np.float32) / np.tanh(float(drive))

# ------------------------------ oscillators -----------------------------------
def osc_saw(freq_arr, sr):
    # Phase-integrated saw with simple wrap (aliased but light/fast)
    phase = np.cumsum(2*np.pi*freq_arr/sr, dtype=np.float32)
    saw = 2.0*(phase/(2*np.pi) - np.floor(0.5 + phase/(2*np.pi)))
    return saw.astype(np.float32)

def osc_square(freq_arr, sr):
    phase = np.cumsum(2*np.pi*freq_arr/sr, dtype=np.float32)
    return np.sign(np.sin(phase)).astype(np.float32)

def osc_sine(freq_arr, sr):
    phase = np.cumsum(2*np.pi*freq_arr/sr, dtype=np.float32)
    return np.sin(phase).astype(np.float32)

# ------------------------------ envelopes -------------------------------------
def adsr(n_on, sr, a=0.005, d=0.12, s=0.65, r=0.25):
    """ADSR safe for very short notes (n_on can be < a+d). Release starts at note-off."""
    a_n = max(1, int(round(a*sr)))
    d_n = max(1, int(round(d*sr)))
    r_n = max(1, int(round(r*sr)))

    env = np.zeros(n_on + r_n, dtype=np.float32)

    # Attack (partial if needed)
    nA = min(a_n, n_on)
    if nA > 0:
        env[:nA] = np.linspace(0.0, 1.0, nA, endpoint=False, dtype=np.float32)
    pos = nA

    # Decay (partial if needed)
    if pos < n_on:
        nD = min(d_n, n_on - pos)
        if nD > 0:
            decay_full = np.linspace(1.0, s, d_n, endpoint=False, dtype=np.float32)
            env[pos:pos+nD] = decay_full[:nD]
            pos += nD

    # Sustain
    if pos < n_on:
        env[pos:n_on] = s
        pos = n_on

    # Release from current level
    rel_level = float(env[n_on - 1]) if n_on > 0 else 0.0
    env[n_on:n_on + r_n] = np.linspace(rel_level, 0.0, r_n, endpoint=False, dtype=np.float32)
    return env

# ------------------------------ instruments -----------------------------------
def synth_tone(freq, dur, sr, *,
               osc='saw', detune_cents=6.0,
               cutoff_start=7000.0, cutoff_end=2200.0,
               a=0.005, d=0.12, s=0.65, r=0.25,
               vibrato_hz=0.0, vibrato_cents=0.0,
               drive=0.0, amp=0.5):
    """Detuned dual-osc -> LP sweep -> ADSR -> optional drive. Returns mono float32."""
    n_on = int(max(1, round(dur*sr)))
    env = adsr(n_on, sr, a=a, d=d, s=s, r=r)
    n_total = len(env)
    t = np.arange(n_total, dtype=np.float32)/sr

    # Vibrato
    if vibrato_hz > 0.0 and vibrato_cents > 0.0:
        vib = np.sin(2*np.pi*vibrato_hz*t).astype(np.float32) * float(vibrato_cents)
        vib_fac = (2.0 ** (vib/1200.0)).astype(np.float32)
    else:
        vib_fac = 1.0

    det = 2.0 ** (detune_cents/1200.0)
    f1 = np.full(n_total, float(freq), dtype=np.float32)
    f2 = np.full(n_total, float(freq*det), dtype=np.float32)
    if isinstance(vib_fac, np.ndarray):
        f1 *= vib_fac; f2 *= vib_fac

    if osc == 'square':
        x1 = osc_square(f1, sr); x2 = osc_square(f2, sr)
    elif osc == 'sine':
        x1 = osc_sine(f1, sr);   x2 = osc_sine(f2, sr)
    else:
        x1 = osc_saw(f1, sr);    x2 = osc_saw(f2, sr)

    raw = (0.6*x1 + 0.4*x2).astype(np.float32)

    # LP sweep
    c_env = np.linspace(float(cutoff_start), float(cutoff_end), n_total, dtype=np.float32)
    y = one_pole_lp(raw, c_env, sr)

    # Drive
    if drive and drive > 0.0:
        y = soft_clip_tanh(y, drive=drive)

    out = (y * env * float(amp)).astype(np.float32)
    return out

def synth_bass_note(freq, dur, sr, amp=0.35):
    """Detuned square -> LP ~400 Hz, short decay."""
    n = int(max(1, dur*sr)); t = np.arange(n, dtype=np.float32)/sr
    sq1 = np.sign(np.sin(2*np.pi*freq*t)).astype(np.float32)
    sq2 = np.sign(np.sin(2*np.pi*freq*1.01*t)).astype(np.float32)
    x = 0.6*sq1 + 0.4*sq2
    env = np.exp(-t*6.0).astype(np.float32)
    x = x * env
    cutoff = np.full(n, 400.0, dtype=np.float32)
    y = one_pole_lp(x, cutoff, sr)
    return (y * float(amp)).astype(np.float32)

# ---- Drums ----
def synth_kick(sr, dur=0.22, amp=0.9):
    n = int(max(1, dur*sr)); t = np.arange(n, dtype=np.float32)/sr
    f_start, f_end, k = 120.0, 45.0, 20.0
    f = f_end + (f_start - f_end) * np.exp(-k*t)
    phase = np.cumsum(2*np.pi*f/sr, dtype=np.float32)
    body = np.sin(phase).astype(np.float32)
    env = np.exp(-t*28.0).astype(np.float32)
    click = (np.exp(-t*2000.0)*0.5).astype(np.float32)
    sig = body*env + click
    return (sig * float(amp)).astype(np.float32)

def synth_snare(sr, dur=0.18, amp=0.4):
    n = int(max(1, dur*sr)); t = np.arange(n, dtype=np.float32)/sr
    noise = (np.random.rand(n).astype(np.float32)*2.0 - 1.0)
    noise = one_pole_hp(noise, 1800.0, sr)
    tone = np.sin(2*np.pi*190.0*t).astype(np.float32) * np.exp(-t*30.0)
    env = np.exp(-t*40.0).astype(np.float32)
    sig = 0.85*noise*env + 0.15*tone
    return (sig * float(amp)).astype(np.float32)

def synth_hat(sr, dur=0.06, amp=0.22, open_=False):
    n = int(max(1, dur*sr)); t = np.arange(n, dtype=np.float32)/sr
    noise = (np.random.rand(n).astype(np.float32)*2.0 - 1.0)
    noise = one_pole_hp(noise, 6500.0, sr)
    decay = 50.0 if open_ else 120.0
    env = np.exp(-t*decay).astype(np.float32)
    sig = noise * env
    return (sig * float(amp)).astype(np.float32)

def synth_tom(sr, base_hz=120.0, dur=0.25, amp=0.45):
    n = int(max(1, dur*sr)); t = np.arange(n, dtype=np.float32)/sr
    f = base_hz + 10*np.exp(-15.0*t)
    phase = np.cumsum(2*np.pi*f/sr, dtype=np.float32)
    tone = np.sin(phase).astype(np.float32)
    tone *= np.exp(-t*12.0).astype(np.float32)
    attack = (np.random.rand(n).astype(np.float32)*2.0 - 1.0)
    attack = one_pole_hp(attack, 3000.0, sr) * np.exp(-t*200.0).astype(np.float32) * 0.2
    return ((tone + attack) * float(amp)).astype(np.float32)

def synth_crash(sr, dur=1.8, amp=0.25):
    n = int(max(1, dur*sr)); t = np.arange(n, dtype=np.float32)/sr
    noise = (np.random.rand(n).astype(np.float32)*2.0 - 1.0)
    noise = one_pole_hp(noise, 4000.0, sr)
    env = np.exp(-t*2.2).astype(np.float32)
    sig = noise * env
    return (sig * float(amp)).astype(np.float32)

# ------------------------------- mixing utils --------------------------------
def mix_mono(stereo, start, sig, pan=0.0):
    """Mix mono into stereo buffer at 'start' with simple pan; clamps to buffer (no extension)."""
    if start >= stereo.shape[1]:
        return stereo
    pan = float(max(-1.0, min(1.0, pan)))
    lg = 0.5*(1.0 - pan); rg = 0.5*(1.0 + pan)
    end = min(start + len(sig), stereo.shape[1])
    seg = sig[:max(0, end - start)]
    stereo[0, start:end] += seg * lg
    stereo[1, start:end] += seg * rg
    return stereo

def stereo_cross_delay_offline(stereo, sr, delay_s=0.24, mix=0.18):
    d = max(1,int(delay_s*sr)); out = stereo.copy()
    if d < out.shape[1]:
        out[0, d:] += mix*stereo[1, :-d]; out[1, d:] += mix*stereo[0, :-d]
    return out

class CrossDelayState:
    """Streaming version of the cross-delay that exactly matches the offline effect."""
    def __init__(self, sr, delay_s=0.24, mix=0.18):
        self.d = max(0, int(round(delay_s*sr)))
        self.mix = float(mix)
        self.rL = np.zeros(self.d, dtype=np.float32) if self.d > 0 else None
        self.rR = np.zeros(self.d, dtype=np.float32) if self.d > 0 else None

    def apply(self, dryL, dryR):
        if self.d == 0:
            return dryL.copy(), dryR.copy()
        n = dryL.shape[0]
        wetL = dryL.copy()
        wetR = dryR.copy()

        # Cross from R->L and L->R
        k = min(self.d, n)
        # Begin of chunk: pull from ring (previous chunk history)
        wetL[:k] += self.mix * self.rR[:k]
        wetR[:k] += self.mix * self.rL[:k]
        # Remainder of chunk: cross-feed inside the same chunk
        if n > self.d:
            wetL[self.d:] += self.mix * dryR[:n - self.d]
            wetR[self.d:] += self.mix * dryL[:n - self.d]

        # Update rings with latest dry tail
        if n >= self.d:
            self.rL[:] = dryL[-self.d:]
            self.rR[:] = dryR[-self.d:]
        else:
            self.rL = np.concatenate([self.rL[n:], dryL]) if self.d > 0 else None
            self.rR = np.concatenate([self.rR[n:], dryR]) if self.d > 0 else None
        return wetL, wetR

def normalize(stereo, headroom_db=3.0):
    peak = float(np.max(np.abs(stereo)))
    if peak <= 0.0: return stereo
    target = 10.0 ** (-headroom_db/20.0)
    return (stereo * (target/peak)).astype(np.float32)

def write_wav_stereo(path, stereo, sr):
    inter = np.vstack([stereo[0], stereo[1]]).T
    maxv = float(np.max(np.abs(inter)))
    if maxv > 1.0: inter = inter / maxv
    data = (inter * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(data.tobytes())

# --------------------------- PyAudio helpers ---------------------------------
def list_output_devices(pa):
    lines = []
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("maxOutputChannels", 0) > 0:
            api = pa.get_host_api_info_by_index(d["hostApi"])["name"]
            lines.append(f"[{i}] {d['name']}  (host API: {api})")
    return lines

def select_output_device(pa, prefer_name_substr="pulse", explicit_index=None, explicit_name=None):
    if explicit_index is not None:
        info = pa.get_device_info_by_index(int(explicit_index))
        if info.get("maxOutputChannels", 0) > 0:
            return int(explicit_index)
        raise RuntimeError(f"Device index {explicit_index} is not an output device.")
    def find_by_name(substr):
        s = substr.lower()
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d.get("maxOutputChannels", 0) > 0 and s in d.get("name","").lower():
                return i
        return None
    if explicit_name:
        idx = find_by_name(explicit_name)
        if idx is not None: return idx
    idx = find_by_name(prefer_name_substr) if prefer_name_substr else None
    if idx is not None: return idx
    try:
        return pa.get_default_output_device_info().get("index", None)
    except Exception:
        pass
    raise RuntimeError("No usable PyAudio output device found.")

def play_with_pyaudio(stereo, sr, frames_per_buffer=2048, device_index=None, device_name=None):
    with capture_stderr() as cap:
        p = pyaudio.PyAudio()
        try:
            out_index = select_output_device(p, "pulse", device_index, device_name)
            stream = p.open(format=pyaudio.paFloat32, channels=2, rate=sr,
                            output=True, output_device_index=out_index,
                            frames_per_buffer=frames_per_buffer)
            n = stereo.shape[1]; pos = 0
            while pos < n:
                end = min(pos + frames_per_buffer, n)
                chunk = np.column_stack((stereo[0, pos:end], stereo[1, pos:end])).ravel().astype(np.float32)
                stream.write(chunk.tobytes()); pos = end
            stream.stop_stream(); stream.close()
            return ""  # success
        except Exception as e:
            return f"{e}\n{cap.data}"
        finally:
            p.terminate()

# ----------------------------- GM instrument map -----------------------------
def program_to_inst(prog):
    """Return a compact instrument 'tag' based on GM program (0..127)."""
    p = int(prog)
    if p in (29, 30):            # Overdriven/Distortion Guitar
        return 'dist_gtr'
    if 24 <= p <= 31:            # Guitars
        return 'clean_gtr'
    if 32 <= p <= 39:            # Basses
        return 'bass'
    if 40 <= p <= 51:            # Solo strings + ensemble
        return 'strings'
    if 80 <= p <= 87:            # Leads (square/saw)
        return 'lead'
    if p == 56:                  # Harp
        return 'pluck'
    return 'piano'               # default

def inst_params(tag, vel=100):
    """Map instrument tag to synth parameters (coarse but musical)."""
    v = max(1, min(127, int(vel)))
    vsc = (v/127.0) ** 1.4  # perceptual-ish velocity curve
    if tag == 'piano':
        return dict(osc='saw', detune_cents=5.0,
                    cutoff_start=5200.0, cutoff_end=900.0,
                    a=0.003, d=0.22, s=0.55, r=0.6, drive=0.0, amp=0.55*vsc)
    if tag == 'clean_gtr':
        return dict(osc='square', detune_cents=4.0,
                    cutoff_start=6000.0, cutoff_end=1800.0,
                    a=0.004, d=0.12, s=0.60, r=0.35, drive=0.0, amp=0.5*vsc)
    if tag == 'dist_gtr':
        return dict(osc='saw', detune_cents=6.0,
                    cutoff_start=7000.0, cutoff_end=2200.0,
                    a=0.004, d=0.10, s=0.70, r=0.42, drive=3.2, amp=0.48*vsc)
    if tag == 'strings':
        return dict(osc='saw', detune_cents=9.0,
                    cutoff_start=7500.0, cutoff_end=2500.0,
                    a=0.045, d=0.40, s=0.90, r=1.6, drive=0.0, amp=0.42*vsc)
    if tag == 'lead':
        return dict(osc='saw', detune_cents=3.0,
                    cutoff_start=6800.0, cutoff_end=1700.0,
                    a=0.006, d=0.10, s=0.65, r=0.45, drive=1.0, amp=0.5*vsc)
    if tag == 'pluck':
        return dict(osc='saw', detune_cents=4.0,
                    cutoff_start=5400.0, cutoff_end=1200.0,
                    a=0.002, d=0.18, s=0.40, r=0.25, drive=0.0, amp=0.5*vsc)
    # fallback
    return dict(osc='saw', detune_cents=6.0,
                cutoff_start=6500.0, cutoff_end=2000.0,
                a=0.005, d=0.12, s=0.65, r=0.35, drive=0.0, amp=0.5*vsc)

# ----------------------------- MIDI -> note events (Option A) -----------------
class NoteEvent:
    __slots__ = ("start","end","note","vel","chan","program","pan","vol","is_drum")
    def __init__(self, start, end, note, vel, chan, program, pan, vol, is_drum):
        self.start = float(start); self.end = float(end); self.note = int(note)
        self.vel = int(vel); self.chan = int(chan); self.program = int(program)
        self.pan = float(pan); self.vol = float(vol); self.is_drum = bool(is_drum)

def parse_midi_to_events(path):
    """Parse MIDI with same-tick bucketing: NOTE-ONs first, then NOTE-OFFs."""
    mid = mido.MidiFile(path)
    tpq = mid.ticks_per_beat
    tempo = 500000  # default 120 BPM (us/qn)

    # Per-channel state
    state = [{
        "program": 0,
        "volume": 1.0,    # CC7 (0..127) normalized
        "pan": 0.0,       # CC10 (-1..+1)
        "sustain": False, # CC64
        "active": {},     # note -> list of starts [(t, vel, vol, pan, prog)]
        "sustained": {}   # note -> list of starts (same as above)
    } for _ in range(16)]

    def cc_to_pan(v):   # 0..127 -> -1 .. +1
        return (max(0,min(127,int(v))) - 64) / 64.0
    def cc_to_amp(v):   # 0..127 -> 0 .. 1
        return max(0.0, min(1.0, float(v)/127.0))

    merged = mido.merge_tracks(mid.tracks)

    abs_ticks = 0
    abs_secs  = 0.0
    last_musical_time = 0.0
    events = []

    bucket = []  # list of (msg, abs_ticks, abs_secs)

    def flush_bucket():
        nonlocal bucket, tempo, last_musical_time
        if not bucket:
            return

        # 1) Apply Program/CC first
        for msg, t_tick, t_sec in bucket:
            if msg.type == 'program_change':
                if 0 <= msg.channel <= 15:
                    state[msg.channel]["program"] = int(msg.program)
            elif msg.type == 'control_change':
                ch = msg.channel
                if msg.control == 7:      state[ch]["volume"]  = cc_to_amp(msg.value)
                elif msg.control == 10:   state[ch]["pan"]     = cc_to_pan(msg.value)
                elif msg.control == 64:   state[ch]["sustain"] = (msg.value >= 64)
                elif msg.control in (120, 123):  # All Sound Off / All Notes Off
                    # Close ALL notes immediately on this channel (active + sustained)
                    for n, starts in list(state[ch]["active"].items()):
                        for st, vel, vol, pan, prog in starts:
                            events.append(NoteEvent(st, t_sec, n, vel, ch, prog, pan, vol, ch==9))
                            last_musical_time = max(last_musical_time, t_sec)
                    state[ch]["active"].clear()
                    for n, starts in list(state[ch]["sustained"].items()):
                        for st, vel, vol, pan, prog in starts:
                            events.append(NoteEvent(st, t_sec, n, vel, ch, prog, pan, vol, ch==9))
                            last_musical_time = max(last_musical_time, t_sec)
                    state[ch]["sustained"].clear()
                elif msg.control == 121:  # Reset All Controllers -> like sustain off + flush sustained
                    if state[ch]["sustain"]:
                        for n, starts in list(state[ch]["sustained"].items()):
                            for st, vel, vol, pan, prog in starts:
                                events.append(NoteEvent(st, t_sec, n, vel, ch, prog, pan, vol, ch==9))
                                last_musical_time = max(last_musical_time, t_sec)
                        state[ch]["sustained"].clear()
                    state[ch]["sustain"] = False

        # 2) NOTE-ONs first at this absolute time
        for msg, t_tick, t_sec in bucket:
            if msg.type == 'note_on' and msg.velocity > 0:
                ch = msg.channel
                lst = state[ch]["active"].setdefault(msg.note, [])
                lst.append((t_sec, msg.velocity, state[ch]["volume"], state[ch]["pan"], state[ch]["program"]))
                last_musical_time = max(last_musical_time, t_sec)

        # 3) NOTE-OFFs (and note_on vel=0)
        for msg, t_tick, t_sec in bucket:
            if (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                ch = msg.channel; is_drum = (ch == 9)
                lst = state[ch]["active"].get(msg.note, [])
                if lst:
                    st, vel, vol, pan, prog = lst.pop(0)
                    if state[ch]["sustain"] and not is_drum:
                        sus = state[ch]["sustained"].setdefault(msg.note, [])
                        sus.append((st, vel, vol, pan, prog))
                    else:
                        events.append(NoteEvent(st, t_sec, msg.note, vel, ch, prog, pan, vol, is_drum))
                        last_musical_time = max(last_musical_time, t_sec)

        # 4) Tempo/meta last (affects *future* deltas)
        for msg, t_tick, t_sec in bucket:
            if msg.type == 'set_tempo':
                tempo = msg.tempo

        bucket = []

    # Walk merged track and bucket by absolute time
    for msg in merged:
        if msg.time > 0:
            abs_ticks += msg.time
            abs_secs  += mido.tick2second(msg.time, tpq, tempo)
            # New absolute time → flush previous bucket
            flush_bucket()
        bucket.append((msg, abs_ticks, abs_secs))
    flush_bucket()

    # Musical end time (fallback to absolute time if no notes)
    t_end = last_musical_time if last_musical_time > 0.0 else abs_secs

    # Flush any hanging notes at end of file to the musical end time
    for ch in range(16):
        for n, starts in state[ch]["active"].items():
            for st, vel, vol, pan, prog in starts:
                events.append(NoteEvent(st, t_end, n, vel, ch, prog, pan, vol, ch==9))
        for n, starts in state[ch]["sustained"].items():
            for st, vel, vol, pan, prog in starts:
                events.append(NoteEvent(st, t_end, n, vel, ch, prog, pan, vol, ch==9))

    return events, t_end

# -------------------------- helpers: progress & preview -----------------------
def render_test_tone(sr=SR, dur=2.0, hz=440.0):
    n = int(dur*sr)
    t = np.arange(n, dtype=np.float32)/sr
    x = np.sin(2*np.pi*hz*t).astype(np.float32) * 0.2
    stereo = np.zeros((2, n), dtype=np.float32)
    stereo[0] += x; stereo[1] += x
    stereo = stereo_cross_delay_offline(stereo, sr, delay_s=0.22, mix=0.18)
    return normalize(stereo, headroom_db=3.0)

def make_progress_printer(total_events):
    def cb(i, t_sec):
        pct = (i + 1) / max(1, total_events) * 100.0
        sys.stderr.write(f"\rRendering: {pct:5.1f}%  @ {t_sec:7.2f}s")
        sys.stderr.flush()
        if i + 1 == total_events:
            sys.stderr.write("\n")
    return cb

def limit_events_to_preview(events, t_preview):
    if not t_preview or t_preview <= 0.0:
        return events
    clipped = []
    for ev in events:
        if ev.start >= t_preview:
            continue
        end = min(ev.end, t_preview)
        clipped.append(
            NoteEvent(ev.start, end, ev.note, ev.vel, ev.chan,
                      ev.program, ev.pan, ev.vol, ev.is_drum)
        )
    return clipped

def apply_polyphony_limit(events, poly_limit=32, window_s=0.020):
    """Drop lowest-velocity pitched notes in 20ms buckets beyond the poly limit."""
    if poly_limit is None or poly_limit <= 0:
        return list(events)
    buckets = {}
    for i, ev in enumerate(events):
        if ev.is_drum:  # drums unbounded
            continue
        b = int(ev.start / window_s)
        buckets.setdefault(b, []).append(i)
    drop = set()
    for b, idxs in buckets.items():
        if len(idxs) > poly_limit:
            idxs_sorted = sorted(idxs, key=lambda k: events[k].vel, reverse=True)
            for k in idxs_sorted[poly_limit:]:
                drop.add(k)
    return [ev for i, ev in enumerate(events) if i not in drop]

# ------------------------------- offline rendering ----------------------------
def render_events(events, sr=SR, poly_limit=32, master_gain=0.9, *,
                  progress=None, cap_end_s=None, tail_s=2.0, max_note_dur_s=30.0):
    """Offline render all events into a stereo buffer. Drums excluded from poly limit."""
    if not events:
        return np.zeros((2, int(sr*1.0)), dtype=np.float32)

    eps = 1.0 / sr
    def capped_end(ev):
        e = ev.end
        if max_note_dur_s and max_note_dur_s > 0.0:
            e = min(e, ev.start + float(max_note_dur_s))
        if cap_end_s is not None:
            e = min(e, float(cap_end_s))
        return e

    # Determine final length
    ends = []
    for ev in events:
        e = capped_end(ev)
        if e - ev.start > eps:
            ends.append(e)
    if not ends:
        n_total = int(sr * 1.0)
    else:
        last_end = max(ends) + float(tail_s)
        n_total = int(max(1, round(last_end * sr)))
    buf = np.zeros((2, n_total), dtype=np.float32)

    # Polyphony limit
    events_use = apply_polyphony_limit(events, poly_limit=poly_limit, window_s=0.020)
    N = len(events_use)

    for i, ev in enumerate(events_use):
        e = capped_end(ev)
        dur_s = e - ev.start
        if dur_s <= eps:
            if progress and (i % 64 == 0 or i + 1 == N):
                progress(i, ev.start)
            continue

        start = int(ev.start * sr)
        if start >= n_total:
            if progress and (i % 64 == 0 or i + 1 == N):
                progress(i, ev.start)
            continue

        # Keep inside buffer
        max_frames_here = n_total - start
        if max_frames_here <= 0:
            continue
        dur_s = min(dur_s, max_frames_here / sr)

        pan = max(-1.0, min(1.0, ev.pan))
        ch_amp = max(0.0, min(1.0, ev.vol))
        vel_amp = (max(1, min(127, ev.vel))/127.0) ** 1.4
        amp_scale = ch_amp * vel_amp

        if ev.is_drum:
            n = ev.note
            if n in (35,36):      sig = synth_kick(sr, 0.22, 0.85 * amp_scale)
            elif n in (38,40):    sig = synth_snare(sr, 0.18, 0.40 * amp_scale)
            elif n in (42,44):    sig = synth_hat(sr, 0.06, 0.22 * amp_scale, open_=False)
            elif n == 46:         sig = synth_hat(sr, 0.22, 0.22 * amp_scale, open_=True)
            elif n in (49,57):    sig = synth_crash(sr, 1.8, 0.25 * amp_scale)
            elif n in (45,47):    sig = synth_tom(sr, 120.0, 0.28, 0.42 * amp_scale)
            elif n in (48,50):    sig = synth_tom(sr, 170.0, 0.26, 0.40 * amp_scale)
            else:                 sig = synth_hat(sr, 0.06, 0.18 * amp_scale, open_=False)
            buf = mix_mono(buf, start, sig, pan=pan)
        else:
            inst = program_to_inst(ev.program)
            if inst == 'bass':
                sig = synth_bass_note(midi_to_freq(ev.note), dur_s, sr, amp=0.38 * amp_scale)
            else:
                prm = inst_params(inst, vel=ev.vel)
                if inst == 'lead':
                    sig = synth_tone(midi_to_freq(ev.note), dur_s, sr,
                                     osc=prm.get('osc','saw'),
                                     detune_cents=prm.get('detune_cents',6.0),
                                     cutoff_start=prm.get('cutoff_start',7000.0),
                                     cutoff_end=prm.get('cutoff_end',2200.0),
                                     a=prm.get('a',0.005), d=prm.get('d',0.12),
                                     s=prm.get('s',0.65), r=prm.get('r',0.35),
                                     vibrato_hz=5.2, vibrato_cents=6.0,
                                     drive=prm.get('drive',0.0), amp=prm.get('amp',0.5)*amp_scale)
                else:
                    sig = synth_tone(midi_to_freq(ev.note), dur_s, sr,
                                     osc=prm.get('osc','saw'),
                                     detune_cents=prm.get('detune_cents',6.0),
                                     cutoff_start=prm.get('cutoff_start',7000.0),
                                     cutoff_end=prm.get('cutoff_end',2200.0),
                                     a=prm.get('a',0.005), d=prm.get('d',0.12),
                                     s=prm.get('s',0.65), r=prm.get('r',0.35),
                                     vibrato_hz=0.0, vibrato_cents=0.0,
                                     drive=prm.get('drive',0.0), amp=prm.get('amp',0.5)*amp_scale)
            buf = mix_mono(buf, start, sig, pan=pan)

        if progress and (i % 64 == 0 or i + 1 == N):
            progress(i, ev.start)

    # FX + normalize + master gain
    buf = stereo_cross_delay_offline(buf, sr, delay_s=0.24, mix=0.16)
    buf = normalize(buf, headroom_db=3.0)
    buf *= float(master_gain)
    return buf.astype(np.float32)

# ------------------------------- realtime engine ------------------------------
class RealtimePreRenderer:
    """Produces stereo chunks with identical synthesis/FX to offline, ahead of playback."""
    def __init__(self, events, sr, frames_per_buffer, *,
                 cap_end_s, tail_s, max_note_dur_s,
                 poly_limit, delay_s=0.24, delay_mix=0.16):
        self.sr = int(sr)
        self.fpb = int(frames_per_buffer)
        self.cap_end_s = float(cap_end_s)
        self.tail_s = float(tail_s)
        self.max_note_dur_s = float(max_note_dur_s) if max_note_dur_s else 0.0
        self.poly_limit = int(poly_limit)
        self.delay = CrossDelayState(sr, delay_s=delay_s, mix=delay_mix)

        # Apply polyphony limiting once (same policy as offline)
        self.events = apply_polyphony_limit(events, poly_limit=self.poly_limit, window_s=0.020)
        self.events.sort(key=lambda e: e.start)

        self.eps = 1.0 / self.sr
        self.note_cache = {}  # idx -> mono array for that event (amp baked-in)
        self.n = len(self.events)

        # Determine total frames to render (cap end + tail)
        last_end = self.cap_end_s + self.tail_s
        self.total_frames = int(max(1, round(last_end * self.sr)))
        self.total_chunks = int((self.total_frames + self.fpb - 1) // self.fpb)

        # For discovery of events intersecting a chunk quickly
        self._event_idx = 0  # index of first event not yet fully past render cursor

    def _capped_end(self, ev):
        e = ev.end
        if self.max_note_dur_s > 0.0:
            e = min(e, ev.start + self.max_note_dur_s)
        e = min(e, self.cap_end_s)
        return e

    def _build_event_signal(self, ev, dur_s, amp_scale):
        """Create and cache per-event mono signal (full dur + release)."""
        if ev.is_drum:
            n = ev.note
            # base shapes (amp per-hit)
            if n in (35,36):      sig = synth_kick(self.sr, 0.22, 0.85 * amp_scale)
            elif n in (38,40):    sig = synth_snare(self.sr, 0.18, 0.40 * amp_scale)
            elif n in (42,44):    sig = synth_hat(self.sr, 0.06, 0.22 * amp_scale, open_=False)
            elif n == 46:         sig = synth_hat(self.sr, 0.22, 0.22 * amp_scale, open_=True)
            elif n in (49,57):    sig = synth_crash(self.sr, 1.8, 0.25 * amp_scale)
            elif n in (45,47):    sig = synth_tom(self.sr, 120.0, 0.28, 0.42 * amp_scale)
            elif n in (48,50):    sig = synth_tom(self.sr, 170.0, 0.26, 0.40 * amp_scale)
            else:                 sig = synth_hat(self.sr, 0.06, 0.18 * amp_scale, open_=False)
            return sig.astype(np.float32)

        inst = program_to_inst(ev.program)
        if inst == 'bass':
            return synth_bass_note(midi_to_freq(ev.note), dur_s, self.sr, amp=0.38 * amp_scale).astype(np.float32)
        else:
            prm = inst_params(inst, vel=ev.vel)
            if inst == 'lead':
                return synth_tone(midi_to_freq(ev.note), dur_s, self.sr,
                                  osc=prm.get('osc','saw'),
                                  detune_cents=prm.get('detune_cents',6.0),
                                  cutoff_start=prm.get('cutoff_start',7000.0),
                                  cutoff_end=prm.get('cutoff_end',2200.0),
                                  a=prm.get('a',0.005), d=prm.get('d',0.12),
                                  s=prm.get('s',0.65), r=prm.get('r',0.35),
                                  vibrato_hz=5.2, vibrato_cents=6.0,
                                  drive=prm.get('drive',0.0), amp=prm.get('amp',0.5)*amp_scale).astype(np.float32)
            else:
                return synth_tone(midi_to_freq(ev.note), dur_s, self.sr,
                                  osc=prm.get('osc','saw'),
                                  detune_cents=prm.get('detune_cents',6.0),
                                  cutoff_start=prm.get('cutoff_start',7000.0),
                                  cutoff_end=prm.get('cutoff_end',2200.0),
                                  a=prm.get('a',0.005), d=prm.get('d',0.12),
                                  s=prm.get('s',0.65), r=prm.get('r',0.35),
                                  vibrato_hz=0.0, vibrato_cents=0.0,
                                  drive=prm.get('drive',0.0), amp=prm.get('amp',0.5)*amp_scale).astype(np.float32)

    def _event_amp_pan(self, ev):
        pan = max(-1.0, min(1.0, ev.pan))
        ch_amp = max(0.0, min(1.0, ev.vol))
        vel_amp = (max(1, min(127, ev.vel))/127.0) ** 1.4
        return ch_amp * vel_amp, pan

    def render_chunk(self, chunk_idx):
        """Render dry->wet chunk [t0, t1). Returns (wetL, wetR, peak_abs)."""
        t0 = (chunk_idx * self.fpb) / self.sr
        t1 = ((chunk_idx + 1) * self.fpb) / self.sr
        # Clamp to total length
        if t0 >= self.total_frames / self.sr:
            n = self.fpb
            return np.zeros(n, np.float32), np.zeros(n, np.float32), 0.0

        n = self.fpb
        dryL = np.zeros(n, dtype=np.float32)
        dryR = np.zeros(n, dtype=np.float32)

        # Advance starting index for events that ended before t0
        while self._event_idx < self.n and self._capped_end(self.events[self._event_idx]) <= t0 + self.eps:
            self._event_idx += 1

        # Mix all events that overlap [t0, t1)
        # Scan backwards a bit to capture long notes that started before t0
        i = max(0, self._event_idx - 1024)
        while i < self.n:
            ev = self.events[i]
            ev_end = self._capped_end(ev)
            if ev.start >= t1 + self.eps:
                break  # future note
            if ev_end <= t0 + self.eps:
                i += 1
                continue  # already finished
            # Overlaps
            amp_scale, pan = self._event_amp_pan(ev)
            # Build (or reuse) full event signal
            key = i
            if key not in self.note_cache:
                dur_s = max(self.eps, ev_end - ev.start)
                self.note_cache[key] = self._build_event_signal(ev, dur_s, amp_scale)
            sig = self.note_cache[key]
            # Compute segment indices within this chunk
            # Note sample index 0 corresponds to ev.start
            src_start = int(round(max(0.0, (t0 - ev.start) * self.sr)))
            dst_start = int(round(max(0.0, (ev.start - t0) * self.sr)))
            n_copy = min(sig.shape[0] - src_start, n - dst_start)
            if n_copy > 0:
                seg = sig[src_start:src_start + n_copy]
                lg = 0.5 * (1.0 - pan); rg = 0.5 * (1.0 + pan)
                dryL[dst_start:dst_start + n_copy] += seg * lg
                dryR[dst_start:dst_start + n_copy] += seg * rg
            i += 1

        # Apply streaming cross-delay (bit-exact with offline)
        wetL, wetR = self.delay.apply(dryL, dryR)

        peak = float(max(np.max(np.abs(wetL)), np.max(np.abs(wetR))))
        return wetL, wetR, peak

# ---------------------- realtime playback with prebuffer ----------------------
class RealtimePlayer:
    def __init__(self, prerenderer: RealtimePreRenderer, *,
                 device_index=None, device_name=None, frames_per_buffer=2048,
                 master_gain=0.90, headroom_db=3.0, prebuffer_s=3.0, progress=False):
        self.r = prerenderer
        self.fpb = int(frames_per_buffer)
        self.master_gain = float(master_gain)
        self.target = 10.0 ** (-float(headroom_db)/20.0)
        self.device_index = device_index
        self.device_name = device_name
        self.prebuffer_s = max(0.0, float(prebuffer_s))
        self.progress = bool(progress)

        # Queues
        self.audio_q = deque()  # holds (wetL, wetR, peak)
        self.lock = threading.Lock()
        self.produced_chunks = 0
        self.done = False
        self.error = None

        # For lookahead normalization
        self.chunk_duration = self.fpb / self.r.sr
        self.prebuffer_chunks = int(np.ceil(self.prebuffer_s / self.chunk_duration))
        self.current_gain = 1.0

    def _producer(self):
        try:
            for ci in range(self.r.total_chunks):
                wetL, wetR, peak = self.r.render_chunk(ci)
                with self.lock:
                    self.audio_q.append((wetL, wetR, peak))
                    self.produced_chunks += 1
                if self.progress and (ci % 32 == 0 or ci + 1 == self.r.total_chunks):
                    buffered = max(0, self.produced_chunks) * self.chunk_duration
                    sys.stderr.write(f"\rPre-rendered: {buffered:7.2f}s / {self.r.total_chunks*self.chunk_duration:7.2f}s")
                    sys.stderr.flush()
            self.done = True
            if self.progress:
                sys.stderr.write("\n")
        except Exception as e:
            self.error = f"Producer error: {e}"

    def play(self):
        with capture_stderr() as cap:
            p = pyaudio.PyAudio()
            try:
                out_index = select_output_device(p, "pulse", self.device_index, self.device_name)
                stream = p.open(format=pyaudio.paFloat32, channels=2, rate=self.r.sr,
                                output=True, output_device_index=out_index,
                                frames_per_buffer=self.fpb)
                # Print device list (stdout) as per "quiet" behavior
                dev_lines = list_output_devices(p)
                for line in dev_lines:
                    print(line)

                # Kick off producer
                th = threading.Thread(target=self._producer, daemon=True)
                th.start()

                # Wait for prebuffer
                while True:
                    with self.lock:
                        ready = len(self.audio_q)
                    if ready >= self.prebuffer_chunks or self.done or self.error:
                        break
                    time.sleep(0.01)

                if self.error:
                    return self.error

                # Playback loop with look-ahead normalization
                chunk_idx = 0
                total_chunks = self.r.total_chunks
                while True:
                    with self.lock:
                        if len(self.audio_q) == 0:
                            if self.done:
                                break
                            # Underflow: wait briefly
                            need_wait = True
                        else:
                            wetL, wetR, peak = self.audio_q.popleft()
                            need_wait = False

                    if need_wait:
                        time.sleep(0.002)
                        continue

                    # Compute look-ahead peak across next N chunks
                    with self.lock:
                        peaks = [pk for (_, _, pk) in list(self.audio_q)[:self.prebuffer_chunks]]
                    future_peak = max([peak] + peaks + [1e-6])
                    target_gain = min(1.0, self.target / future_peak) * self.master_gain

                    # Smooth ramp within the chunk to avoid steps
                    n = wetL.shape[0]
                    ramp = np.linspace(self.current_gain, target_gain, n, dtype=np.float32)
                    wetL *= ramp; wetR *= ramp
                    self.current_gain = float(target_gain)

                    # Interleave and send
                    interleaved = np.column_stack((wetL, wetR)).ravel().astype(np.float32)
                    stream.write(interleaved.tobytes())
                    chunk_idx += 1

                    if self.progress and (chunk_idx % 64 == 0 or chunk_idx == total_chunks):
                        played = chunk_idx * self.chunk_duration
                        sys.stderr.write(f"\rPlaying (RT): {played:7.2f}s / {total_chunks*self.chunk_duration:7.2f}s")
                        sys.stderr.flush()

                stream.stop_stream(); stream.close()

                if self.progress:
                    sys.stderr.write("\n")
                return ""  # success

            except Exception as e:
                return f"{e}\n{cap.data}"
            finally:
                p.terminate()

# ----------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="32‑voice MIDI softsynth with PyAudio playback (quiet).")
    ap.add_argument("--midi", help="Path to a legally obtained MIDI file to render.")
    ap.add_argument("--frames", type=int, default=2048, help="Frames per buffer for streaming (PyAudio).")
    ap.add_argument("--device-index", type=int, default=None, help="PyAudio output device index.")
    ap.add_argument("--device-name", type=str, default=None, help="Substring to select device (e.g., 'pulse').")
    ap.add_argument("--poly", type=int, default=32, help="Max pitched voices per 20ms window (drums excluded).")
    ap.add_argument("--gain", type=float, default=0.90, help="Master gain multiplier after normalization.")
    ap.add_argument("--save", action="store_true", help="Also save rendered WAV (midi_synth32.wav).")
    ap.add_argument("--progress", action="store_true", help="Show progress on stderr.")
    ap.add_argument("--preview", type=float, default=0.0, help="Render only the first N seconds (0 = full song).")
    ap.add_argument("--test-tone", action="store_true", help="Play a 2s test tone instead of MIDI.")
    ap.add_argument("--tail", type=float, default=2.0, help="Extra seconds after the last musical event.")
    ap.add_argument("--max-note-dur", type=float, default=30.0, help="Clamp any single note duration (0 disables).")
    # New realtime options
    ap.add_argument("--realtime", action="store_true", help="Play in realtime with prebuffered chunks (no DSP simplifications).")
    ap.add_argument("--prebuffer", type=float, default=3.0, help="Seconds of audio to prebuffer/look-ahead in realtime mode.")
    args = ap.parse_args()

    # Print ONLY the device list (stdout) immediately (even before MIDI) so you see life.
    with capture_stderr():
        pa = pyaudio.PyAudio()
        try:
            dev_lines = list_output_devices(pa)
        finally:
            pa.terminate()
    for line in dev_lines:
        print(line)

    if args.test_tone:
        buf = render_test_tone(sr=SR, dur=2.0, hz=440.0)
        err = play_with_pyaudio(buf, SR, frames_per_buffer=args.frames,
                                device_index=args.device_index, device_name=args.device_name)
        if err:
            sys.stderr.write(err); sys.exit(1)
        return

    if not args.midi:
        sys.stderr.write("error: --midi is required unless --test-tone is used\n")
        sys.exit(2)

    # Parse MIDI with same-tick NOTE-ON-before-NOTE-OFF policy (Option A)
    events, music_end = parse_midi_to_events(args.midi)

    if args.realtime:
        # Realtime path: render chunk-by-chunk at full quality with prebuffer look-ahead
        cap_end = music_end  # full musical end
        prer = RealtimePreRenderer(
            events, SR, frames_per_buffer=args.frames,
            cap_end_s=cap_end, tail_s=float(args.tail),
            max_note_dur_s=(float(args.max_note_dur) if float(args.max_note_dur) > 0.0 else 0.0),
            poly_limit=int(args.poly), delay_s=0.24, delay_mix=0.16
        )
        player = RealtimePlayer(
            prer, device_index=args.device_index, device_name=args.device_name,
            frames_per_buffer=args.frames, master_gain=float(args.gain),
            headroom_db=3.0, prebuffer_s=float(args.prebuffer), progress=args.progress
        )
        err = player.play()
        if err:
            sys.stderr.write(err); sys.exit(1)
        return

    # Offline render path (previous behavior)
    t_prev = args.preview if args.preview and args.preview > 0.0 else 0.0
    events_to_render = limit_events_to_preview(events, t_prev) if t_prev > 0.0 else events
    cap_end = min(music_end, t_prev) if t_prev > 0.0 else music_end
    progress_cb = make_progress_printer(len(events_to_render)) if args.progress else None

    buf = render_events(
        events_to_render, sr=SR,
        poly_limit=int(args.poly),
        master_gain=float(args.gain),
        progress=progress_cb,
        cap_end_s=cap_end,
        tail_s=float(args.tail),
        max_note_dur_s=(float(args.max_note_dur) if float(args.max_note_dur) > 0.0 else 0.0)
    )

    # Play; on error, print captured stderr + exception
    err = play_with_pyaudio(buf, SR, frames_per_buffer=args.frames,
                            device_index=args.device_index, device_name=args.device_name)
    if err:
        sys.stderr.write(err)
        sys.exit(1)

    if args.save:
        write_wav_stereo("midi_synth32.wav", buf, SR)

if __name__ == "__main__":
    main()
