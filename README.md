# soundtest
Routines to test sound output under linux or WSL (C/Python)
# Synth Intro (Python)

This small project plays a **polyphonic**, synthy intro. It generates a detuned-saw pad under an arpeggio across the Em–C–G–D progression and plays on the **default audio device**.
## Quick start (venv)


### 0) Set up audio packages (on any debian-based linux and WSL)
```bash
sudo apt update
sudo apt install -y pulseaudio-utils sound-theme-freedesktop \
    portaudio19-dev python3-dev build-essential pkg-config \
    sox ffmpeg

```
### 1) Create & activate a virtual environment
```bash
python3 -m venv soundtest
source soundtest/bin/activate
```
### 2) Upgrade pip (recommended) and install dependencies
```bash
export CFLAGS="$CFLAGS -I/usr/include/portaudio2"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```
### 3) Run it
```bash
python synth_intro.py
```