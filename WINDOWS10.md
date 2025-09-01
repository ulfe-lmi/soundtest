
# Audio Setup in WSL2 on Windows 10 (with GWSL)

Windows 10 does **not** support WSLg (which provides built-in audio on Windows 11).  
To enable audio in **WSL2 on Windows 10**, you must use a third-party X server and PulseAudio server — the simplest option is **[GWSL](https://opticos.github.io/gwsl/)**.

These steps explain how to set up **PulseAudio-only** sound in WSL2 (no ALSA devices needed).

---

## 1. Install GWSL on Windows 10
- Download and install [GWSL from the Microsoft Store](https://apps.microsoft.com/detail/9NL6KD1H33V3).
- Launch GWSL from the Start menu; it will run an X server and a PulseAudio server in the background.

---

## 2. Configure environment variables in WSL2
Edit your shell startup file (`~/.bashrc` or `~/.zshrc`) and add:

```bash
# Tell Linux apps where to find the Windows X server (for GUI apps)
export DISPLAY=$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf):0.0

# Tell Linux apps where to send audio (to GWSL PulseAudio server)
export PULSE_SERVER=tcp:$(awk '/nameserver/ {print $2; exit}' /etc/resolv.conf) #GWSL
```

Reload your config:
```bash
source ~/.bashrc
```

---

## 3. Install PulseAudio client tools in Ubuntu
Inside WSL2 (Ubuntu/Debian/etc.):
```bash
sudo apt update
sudo apt install -y pulseaudio-utils
```

This gives you the `paplay` tool for testing.

---

## 4. Test audio playback
Play a test sound (make sure GWSL is running in Windows first):

```bash
paplay /usr/share/sounds/freedesktop/stereo/bell.oga
```

You should hear a “bell” from your Windows speakers.

---

## 5. Using Python or other apps
Most modern libraries can talk directly to PulseAudio. For example, in Python with `sounddevice`:

```python
import sounddevice as sd
print(sd.query_devices())   # Look for 'pulse'
sd.default.device = 'pulse'
```

Then playback will go through GWSL’s PulseAudio server.

---

## ✅ Summary
- **Windows 11:** WSLg has native audio, no setup needed.  
- **Windows 10:** Must use **GWSL (or X410/VcXsrv + PulseAudio)**.  
- Environment variable `PULSE_SERVER=tcp:...` is required.  
- Use `pulseaudio-utils` and `paplay` to test sound.  
- No ALSA devices are present in WSL2 → stick with PulseAudio.

Now your Linux apps (including Python scripts) can play sound through WSL2 on Windows 10!
