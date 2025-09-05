import argparse


def get_parser() -> argparse.ArgumentParser:
    """Build and return the command line argument parser."""
    ap = argparse.ArgumentParser(
        description="32‑voice MIDI softsynth with PyAudio playback (quiet)."
    )
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
    ap.add_argument(
        "--max-note-dur", type=float, default=30.0,
        help="Clamp any single note duration (0 disables)."
    )
    ap.add_argument(
        "--realtime", action="store_true",
        help="Play in realtime with prebuffered chunks (no DSP simplifications)."
    )
    ap.add_argument(
        "--prebuffer", type=float, default=3.0,
        help="Seconds of audio to prebuffer/look-ahead in realtime mode."
    )
    return ap
