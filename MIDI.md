# 32-Voice MIDI Softsynth (`midi_synth32.py`)

A tiny, dependency-light MIDI renderer that plays or turns a standard `.mid` file into audio using a simple subtractive synth and a GM-style drum kit. It’s designed to be **quiet** on the command line (prints only your audio device list on success), robust with real-world MIDI (sustain, tempo, “all notes off”), and easy to tweak.

---

## NEW in this version
- **Realtime playback**: `--realtime` streams audio in sync with **identical synthesis/FX** as offline.  
  Use `--prebuffer <seconds>` to pre-render a look-ahead queue so playback never starves.
- **Same-tick bucketing**: parser processes **NOTE-ONs before NOTE-OFFs** at the **same absolute time**, preventing “never-ending notes” in files that re-strike on the same tick.
- **Safety guards**: honors **CC120/123/121**, short-note-safe ADSR, clamped mixing, musical end detection, `--tail`, `--max-note-dur`.
- **Streaming FX**: cross-delay is applied streamingly so the realtime result matches the offline effect.

---

## Quick demo (download + play)
1. Download the demo MIDI from: **https://onlinesequencer.net/3446742** (use *Download MIDI* on that page).  
2. Then run (replace `[filename].mid` with your downloaded file name):
   ```bash
   python midi_synth32.py --midi [filename].mid --realtime --prebuffer 5 --gain 10
   ```
   > Tip: `--gain 10` is intentionally hot. If your interface distorts, reduce to `--gain 1.5` or `--gain 2`.

---

## Features at a glance
- **Polyphony:** up to 32 pitched voices (configurable with `--poly`), drums excluded from the limiter.
- **MIDI support:** tempo changes, Program Change, CC7 (volume), CC10 (pan), CC64 (sustain), CC120 (All Sound Off), CC123 (All Notes Off), CC121 (Reset All Controllers).
- **GM mapping (compact):** piano, clean/distortion guitars, strings, bass, simple lead; channel 10 (index 9) = drums.
- **Synthesis:** dual detuned oscillators → one-pole low-pass sweep → ADSR → optional soft clip; noise-based drum synths.
- **FX & gain staging:** light stereo cross-delay for width, peak normalization with headroom, master gain.
- **CLI quality-of-life:** progress bar to `stderr`, preview window, test-tone generator, device selection.
- **Safety:** short-note-safe ADSR, caps on note length and tail so songs don’t “ring forever.”

---

## Requirements
- Python 3.8+
- Packages: `numpy`, `pyaudio`, `mido`
- A working system audio backend supported by PortAudio (PyAudio’s engine)

Install:
```bash
pip install numpy pyaudio mido
```

> **WSL2 note:** You’ll need an audio bridge (e.g., PulseAudio/PipeWire to Windows). Selecting a device by name with `--device-name pulse` often helps. See **Troubleshooting** below.

---

## Quick start

### Offline (render-then-play)
```bash
python midi_synth32.py --midi song.mid --progress
python midi_synth32.py --midi song.mid --save --progress
```

### Realtime (no compromises on tone)
```bash
python midi_synth32.py --midi song.mid --realtime --prebuffer 3 --progress
```
- `--prebuffer` pre-renders N seconds ahead and gives the look-ahead window used for clean, non-clipping normalization.
- If you hear dropouts, increase `--frames` (e.g., `4096`) and/or `--prebuffer` (e.g., `5–8`).

### Audio path sanity check
```bash
python midi_synth32.py --test-tone
```

On success, **stdout** prints only your audio device list (one line per device). Any progress or errors go to **stderr**.

---

## Command-line options

- `--midi PATH` — MIDI file to render. Required unless `--test-tone` is used.
- `--frames INT` — PyAudio buffer size (default `2048`). Larger is safer; smaller reduces latency.
- `--device-index INT` — Force a specific output device by index.
- `--device-name SUBSTR` — Pick the first device whose name contains this substring (e.g., `pulse`, `WASAPI`).
- `--poly INT` — Max simultaneous pitched voices per 20 ms window (default `32`). Drums aren’t limited.
- `--gain FLOAT` — Master gain after normalization (default `0.90`).
- `--save` — Also write `midi_synth32.wav` next to the script.
- `--progress` — Print a simple render progress indicator to `stderr`.
- `--preview SECONDS` — Only render the first N seconds (speeds up testing).
- `--test-tone` — Play a 2 s 440 Hz tone; skips MIDI entirely.
- `--tail SECONDS` — Extra decay after the last **musical** event (default `2.0`).
- `--max-note-dur SECONDS` — Clamp any single note (default `30.0`; `0` disables).
- **Realtime-only:**
  - `--realtime` — Stream audio with full-quality synthesis and streaming FX.
  - `--prebuffer SECONDS` — Seconds of audio to pre-render and to use for look-ahead normalization (default `3.0`).

Examples:
```bash
# Choose device by name (good on WSL2 / Linux PulseAudio)
python midi_synth32.py --midi song.mid --device-name pulse --progress

# Use an explicit device index and a bigger buffer
python midi_synth32.py --midi song.mid --device-index 3 --frames 4096

# Heavier project? Keep it safe with fewer voices
python midi_synth32.py --midi song.mid --poly 20 --progress
```

---

## How it works (pipeline)

1) **Parse MIDI → events**  
   *Tracks are merged; absolute time accumulates using the current tempo.*  
   - **Same-tick bucketing**: messages at the same absolute time are handled in buckets. Inside each bucket:  
     **Program/CC → NOTE-ONs → NOTE-OFFs → Tempo/meta.**  
     This ensures off/on re-strikes at the same tick don’t create “stuck” notes.  
   - Per-channel state tracks Program, Volume (CC7), Pan (CC10), Sustain (CC64).  
   - `CC64` keeps pitched notes in a **sustained** pool; on pedal release they’re closed at that moment.  
   - `CC123` (All Notes Off), `CC120` (All Sound Off), `CC121` (Reset Controllers) immediately close active/sustained notes.  
   - The end of the piece is computed from the **last musical time** (any note/drum), not the last meta-event.

2) **Event capping & render length**  
   - Each event’s end is capped by `--max-note-dur` (if > 0) and the musical end; a configurable `--tail` is added.  
   - The stereo buffer is sized once; signals are **clamped** into it (no accidental buffer growth).

3) **Polyphony limiting**  
   - In 20 ms windows, if pitched notes exceed `--poly`, the renderer keeps the loudest by velocity and drops the rest.  
   - Drums do not count against `--poly`.

4) **Synthesis**  
   - **Pitched:** dual detuned oscillators (saw/square/sine) → one-pole LP sweep (cutoff from `cutoff_start` to `cutoff_end`) → ADSR → optional soft clip.  
   - **Drums:** procedural (kick, snare, hi-hat open/closed, toms, crash) using noise/highpass, short FM-ish tones, and shaped decays.  
   - **FX & mix:** a light stereo cross-delay for width; peak normalization to ~−3 dB headroom; master gain.

---

## Realtime mode (identical quality)

- The renderer pre-builds audio **chunk-by-chunk** into a queue while PyAudio plays previous chunks.  
- **DSP parity:** pitched voices and drums use the *same* oscillators, filter sweep, ADSR, soft-clip, and the **same** cross-delay (streaming state matches offline).  
- **Look-ahead normalization:** for each chunk, gain ramps toward a target computed from the peak across the next `--prebuffer` seconds, preventing clipping without pumping.  
- For bit-for-bit identical loudness to a full offline pass, perform a peak-measurement pass first (possible extension); in practice, a `--prebuffer` of **3–8 s** sounds transparent.

**If you hear glitches in realtime:** increase `--frames` (e.g., `4096`) and/or `--prebuffer` (e.g., `5–8`), or reduce `--poly` slightly on very dense files.

---

## Sound design (GM tag → synth params)
A compact map chooses per-program “tags”:

- **piano** (default), **clean_gtr**, **dist_gtr**, **strings**, **lead**, **pluck**, **bass**  
- Each tag sets oscillator type, detune amount, filter sweep range, ADSR, drive level, base amp.  
- Loudness follows a perceptual-ish curve of note velocity × channel volume.

> Want a darker piano or grittier guitars? Tweak the `inst_params(...)` values:  
> Piano brightness → lower `cutoff_start` or raise `cutoff_end`.  
> Distortion amount → raise `drive` for `dist_gtr`.  
> String pad length → raise `r` (release).

---

## Performance & quality tips
- **Speeding up tests:** use `--preview 20 --progress` (offline).  
- **Reduce CPU / memory:** drop `--poly` (e.g., 24), or increase `--frames`.  
- **Avoid runaway tails:** keep `--tail` small (2–4 s) and keep `--max-note-dur` sane (e.g., 20–45 s).  
- **Aliasing:** oscillators are naive (fast). For cleaner highs, lower `SR` or filter more aggressively (raise `cutoff_end` slope or add a cheap LP after the ADSR).  
- **Loudness:** if you push `--gain` high (e.g., `10`), be mindful of downstream clipping.

---

## Troubleshooting
**No sound at all**  
Run the test tone:
```bash
python midi_synth32.py --test-tone
```
If you still hear nothing, select a device:
```bash
python midi_synth32.py --test-tone --device-name pulse
# or
python midi_synth32.py --test-tone --device-index 3
```

**WSL2 specifics**  
Ensure PulseAudio/PipeWire is running and bridged to Windows audio. Try `--device-name pulse` or `--device-name wasapi`.  
The script suppresses most C-level noise; genuine errors still appear on `stderr`.

**It “hangs” or takes very long**  
Use `--progress`. Keep `--tail` small (e.g., `2`) and set `--max-note-dur` to a reasonable cap (e.g., `20`).  
The parser already prevents “never-ending” notes via same-tick bucketing.

**Crackles / glitches in realtime**  
Increase `--frames` (e.g., `4096`) and/or `--prebuffer` (e.g., `5–8`). Close heavy background apps.  
If a specific file is extremely dense, reduce `--poly` a touch.

**Device confusion**  
The script prints the **device list** on success. Pick a clear index or use `--device-name`.

---

## Known limitations
- **Pitch bend** (`pitchwheel`) not rendered in this base file (straightforward to add).
- **Modulation / expression:** CC1, CC11, aftertouch ignored.
- **No SysEx / RPN/NRPN.**
- **Single LP per note:** no resonance, no multi-pole filters (by design for speed).
- **FX:** only the cross-delay (no reverb/chorus—can be added).

---

## Legal & usage notes
- This tool **does not include** any copyrighted music or transcriptions.  
- Use it with **legally obtained** MIDI files.  
- The GM mapping and synthesis are intentionally generic; they won’t replicate any particular record’s exact tone.

---

## Changelog (high level)
- **Current:** Added **realtime** mode with `--realtime` and `--prebuffer` (identical synthesis/FX; streaming cross-delay; look-ahead normalization). Implemented **same-tick bucketing** (NOTE-ON before NOTE-OFF) to eliminate “never-ending notes”. Kept CC120/123/121 handling, short-note-safe ADSR, clamped mixing, musical end detection, preview/progress/test-tone, quiet I/O.
