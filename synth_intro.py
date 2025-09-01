#!/usr/bin/env python3
# synth_intro.py — PyAudio-only playback, quiet output (prints only device list on success).
# Adds a simple drum groove (kick/snare/hat) and a bassline on top of the pad + arpeggio.

import argparse
import os
import sys
import tempfile
import numpy as np
import pyaudio
import wave

SR = 44100
TEMPO = 100  # BPM

# -------------------- silence C-level stderr safely (no ctypes) --------------------
class capture_stderr:
    """Temporarily capture fd(2) so ALSA/PortAudio noise is hidden unless there is an error."""
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

# ------------------------------ synth primitives ------------------------------
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

# ------------------------------ musical voices --------------------------------
def synth_note(freq, dur, sr, detune_cents=7.0, cutoff_start=8000.0, cutoff_end=2500.0, amp=0.5, pad=False):
    n_on = int(dur * sr); r_time = 0.50 if pad else 0.22
    n_total = n_on + int(r_time * sr); t = np.arange(n_total, dtype=np.float32) / sr
    detune = 2 ** (detune_cents / 1200.0)
    phase1 = 2*np.pi*freq*t; phase2 = 2*np.pi*(freq*detune)*t
    saw1 = 2.0*(phase1/(2*np.pi) - np.floor(0.5 + phase1/(2*np.pi)))
    saw2 = 2.0*(phase2/(2*np.pi) - np.floor(0.5 + phase2/(2*np.pi)))
    raw = 0.6*saw1 + 0.4*saw2
    c_env = np.linspace(cutoff_start, cutoff_end, n_total, dtype=np.float32)
    filtered = one_pole_lp(raw.astype(np.float32), c_env, sr)
    if pad:  a,d,s,r = 0.03,0.20,0.85,r_time
    else:    a,d,s,r = 0.005,0.12,0.65,r_time
    a_n = max(1,int(a*sr)); d_n = max(1,int(d*sr)); r_n = max(1,int(r*sr))
    sustain = max(0, n_on - a_n - d_n)
    env = np.zeros(n_total, dtype=np.float32)
    env[:a_n] = np.linspace(0.0,1.0,a_n,endpoint=False,dtype=np.float32)
    env[a_n:a_n+d_n] = np.linspace(1.0,s,d_n,endpoint=False,dtype=np.float32)
    env[a_n+d_n:a_n+d_n+sustain] = s
    start_r = a_n + d_n + sustain
    env[start_r:start_r+r_n] = np.linspace(s,0.0,r_n,endpoint=False,dtype=np.float32)
    return (filtered*env*float(amp)).astype(np.float32)

def synth_bass_note(freq, dur, sr, amp=0.35):
    """Solid bass: detuned square -> LP @ ~400 Hz, short-ish decay."""
    n = int(dur*sr); t = np.arange(n, dtype=np.float32)/sr
    sq1 = np.sign(np.sin(2*np.pi*freq*t)).astype(np.float32)
    sq2 = np.sign(np.sin(2*np.pi*freq*1.01*t)).astype(np.float32)
    x = 0.6*sq1 + 0.4*sq2
    env = np.exp(-t*6.0).astype(np.float32)  # quick decay
    x = x * env
    cutoff = np.full(n, 400.0, dtype=np.float32)
    y = one_pole_lp(x, cutoff, sr)
    return (y * float(amp)).astype(np.float32)

def synth_kick(sr, dur=0.2, amp=0.9):
    n = int(dur*sr); t = np.arange(n, dtype=np.float32)/sr
    f_start, f_end, k = 120.0, 45.0, 20.0
    f = f_end + (f_start - f_end) * np.exp(-k*t)
    phase = np.cumsum(2*np.pi*f/sr, dtype=np.float32)
    body = np.sin(phase).astype(np.float32)
    env = np.exp(-t*28.0).astype(np.float32)
    click = (np.exp(-t*2000.0)*0.5).astype(np.float32)
    sig = body*env + click
    return (sig * float(amp)).astype(np.float32)

def synth_snare(sr, dur=0.18, amp=0.35):
    n = int(dur*sr); t = np.arange(n, dtype=np.float32)/sr
    noise = (np.random.rand(n).astype(np.float32)*2.0 - 1.0)
    noise = one_pole_hp(noise, 1800.0, sr)
    tone = np.sin(2*np.pi*190.0*t).astype(np.float32) * np.exp(-t*30.0)
    env = np.exp(-t*40.0).astype(np.float32)
    sig = 0.85*noise*env + 0.15*tone
    return (sig * float(amp)).astype(np.float32)

def synth_hat(sr, dur=0.06, amp=0.18):
    n = int(dur*sr); t = np.arange(n, dtype=np.float32)/sr
    noise = (np.random.rand(n).astype(np.float32)*2.0 - 1.0)
    noise = one_pole_hp(noise, 6000.0, sr)
    env = np.exp(-t*120.0).astype(np.float32)
    sig = noise * env
    return (sig * float(amp)).astype(np.float32)

# ------------------------------- mixing utils --------------------------------
def mix_mono(stereo, start, sig, pan=0.0):
    """Mix a mono signal into stereo buffer at sample index 'start' with simple pan (-1..+1)."""
    pan = float(max(-1.0, min(1.0, pan)))
    lg = 0.5*(1.0 - pan); rg = 0.5*(1.0 + pan)
    end = start + len(sig)
    if end > stereo.shape[1]:
        stereo[:] = np.pad(stereo, ((0,0),(0,end - stereo.shape[1])), mode="constant")
    stereo[0, start:end] += sig * lg
    stereo[1, start:end] += sig * rg
    return stereo

def add_note(buffer, start_sample, freq, dur, amp=0.5, pad=False, pan=0.0):
    sig = synth_note(freq, dur, SR, amp=amp, pad=pad)
    return mix_mono(buffer, start_sample, sig, pan)

def stereo_cross_delay(stereo, sr, delay_s=0.26, mix=0.22):
    d = max(1,int(delay_s*sr)); out = stereo.copy()
    out[0, d:] += mix*stereo[1, :-d]; out[1, d:] += mix*stereo[0, :-d]
    return out

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

# ------------------------------- arrangement ---------------------------------
def build_buffer():
    beat = 60.0 / TEMPO
    bar = 4.0 * beat

    # Harmony: Em, C, G, D triads
    chords = [
        [52, 55, 59],  # E3, G3, B3 (Em)
        [48, 52, 55],  # C3, E3, G3
        [43, 47, 50],  # G2, B2, D3
        [50, 54, 57],  # D3, F#3, A3
    ]
    repeats = 2  # total 8 bars

    total_bars = len(chords) * repeats
    total_len = int((total_bars * bar + 1.0) * SR)  # + tail
    buf = np.zeros((2, total_len), dtype=np.float32)

    # --- pad + arp (slightly lower levels to make room for drums/bass)
    for r in range(repeats):
        for idx, chord in enumerate(chords):
            start = int((r * len(chords) + idx) * bar * SR)
            # pad layer
            for n, m in enumerate(chord):
                pan = -0.3 if n == 0 else (0.3 if n == 2 else 0.0)
                buf = add_note(buf, start, midi_to_freq(m), bar * 0.98, amp=0.18, pad=True, pan=pan)
            # arpeggio: 8th-note pattern
            step = beat / 2.0
            arp_order = [0, 2, 1, 2, 0, 2, 1, 2]
            for s, ord_idx in enumerate(arp_order):
                st = start + int(s * step * SR)
                dur = step * 0.9
                m = chord[ord_idx]
                pan = -0.12 if (s % 2 == 0) else 0.12
                buf = add_note(buf, st, midi_to_freq(m), dur, amp=0.22, pad=False, pan=pan)

    # --- drums (4/4: kick on 1 & 3, snare on 2 & 4, hats on 8ths)
    kick = synth_kick(SR, 0.2, 0.9)
    snare = synth_snare(SR, 0.18, 0.35)
    hat = synth_hat(SR, 0.06, 0.18)

    for bar_idx in range(total_bars):
        bar_start = int(bar_idx * bar * SR)
        # kick on beats 1 and 3
        for beat_idx in (0, 2):
            buf = mix_mono(buf, bar_start + int(beat_idx * beat * SR), kick, pan=0.0)
        # snare on beats 2 and 4
        for beat_idx in (1, 3):
            buf = mix_mono(buf, bar_start + int(beat_idx * beat * SR), snare, pan=0.0)
        # closed hats on 8th notes
        for h in range(8):
            hat_start = bar_start + int((h * (beat / 2.0)) * SR)
            buf = mix_mono(buf, hat_start, hat, pan=0.0)

    # --- bassline (8th-note pattern: root, octave, fifth, octave, ...)
    for r in range(repeats):
        for idx, chord in enumerate(chords):
            start = int((r * len(chords) + idx) * bar * SR)
            root = chord[0] - 12  # drop an octave for bass
            fifth = root + 7
            octave = root + 12
            bass_seq = [root, octave, fifth, octave, root, octave, fifth, octave]
            step = beat / 2.0
            for s, m in enumerate(bass_seq):
                st = start + int(s * step * SR)
                dur = step * 0.9
                sig = synth_bass_note(midi_to_freq(m), dur, SR, amp=0.33)
                buf = mix_mono(buf, st, sig, pan=0.0)

    # space & glue
    buf = stereo_cross_delay(buf, SR, delay_s=0.22, mix=0.18)
    return normalize(buf, headroom_db=3.0)

# ----------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Synth intro via PyAudio + drums + bass; quiet output.")
    ap.add_argument("--save", action="store_true", help="Also save liquido_intro.wav (no extra prints).")
    ap.add_argument("--device-index", type=int, default=None, help="Use a specific PyAudio output device index.")
    ap.add_argument("--device-name", type=str, default=None, help="Substring to select device (e.g., 'pulse').")
    ap.add_argument("--frames", type=int, default=2048, help="Frames per buffer.")
    args = ap.parse_args()

    # Build full mix
    buf = build_buffer()

    # Print ONLY the device list (quiet)
    with capture_stderr():
        pa = pyaudio.PyAudio()
        try:
            dev_lines = list_output_devices(pa)
        finally:
            pa.terminate()
    for line in dev_lines:
        print(line)

    # Play quietly; on error, print captured stderr + exception
    err = play_with_pyaudio(buf, SR, frames_per_buffer=args.frames,
                            device_index=args.device_index, device_name=args.device_name)
    if err:
        sys.stderr.write(err)
        sys.exit(1)

    if args.save:
        write_wav_stereo("synth_intro.wav", buf, SR)

if __name__ == "__main__":
    main()
