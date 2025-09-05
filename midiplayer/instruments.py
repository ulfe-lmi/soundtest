import numpy as np

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

