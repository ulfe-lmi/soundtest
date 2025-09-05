#!/usr/bin/env python3
import argparse
import numpy as np
import pyaudio
import sys
import os

# --------------------------- Argparser ---------------------------
ap = argparse.ArgumentParser(description="FIFO decoder (8-bit int8 payload) with PortAudio device selection.")
ap.add_argument("--fifo", default="audiofifo1.fifo", help="Path to named pipe/FIFO.")
ap.add_argument("--device-name", default="pulse",
                help="Substring to match output device name or host API (e.g., 'pulse', 'jack', 'pipewire').")
ap.add_argument("--device-index", type=int, default=None,
                help="Explicit PyAudio device index to use (overrides --device-name).")
ap.add_argument("--sample-rate", type=int, default=44100, help="Sample rate (Hz).")
ap.add_argument("--channels", type=int, default=2, help="Number of channels.")
ap.add_argument("--block", type=int, default=4096, help="Samples per channel per block.")
ap.add_argument("--bits", type=int, default=8,
                help="(Optional) default bit width if header missing (not used if header present).")
args = ap.parse_args()

fifo_path = args.fifo
sample_rate = float(args.sample_rate)
num_channels = float(args.channels)
samples_per_channel = float(args.block)

# --------------------------- Decode (iz tvoje različice) ---------------------------
def decode(encoded_block):
    """
    Tvoj format: prvi bajt = bit_width (1..8), preostanek = int8 interleaved stereo vzorci.
    (Ta varianta NE uporablja max_coeff – to je skladno z eno od tvojih 8-bitnih poti.):contentReference[oaicite:1]{index=1}
    """
    if not encoded_block:
        return np.zeros(int(samples_per_channel * num_channels), dtype=np.int16)

    bit_width = encoded_block[0]
    # varnost: če bi kdaj prišel header brez bit_width, pade nazaj na args.bits
    if bit_width == 0:
        bit_width = max(1, min(16, args.bits))

    # int8 stereo podatki (preostanek)
    try:
        audio_data = np.frombuffer(encoded_block[1:], dtype=np.int8).reshape(-1, 2)
    except ValueError:
        # če dolg blok ne ustreza, vrni tišino te dolžine
        n_samps = max(0, len(encoded_block) - 1) // 2
        return np.zeros(n_samps * 2, dtype=np.int16)

    # dvig v 16-bit obseg glede na bit width
    scaling_factor = 2 ** (16 - bit_width)
    decoded = (audio_data.astype(np.int16) * scaling_factor).flatten()
    return decoded

# --------------------------- PortAudio izbor naprave ---------------------------
def list_output_devices(pa):
    devs = []
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("maxOutputChannels", 0) > 0:
            api = pa.get_host_api_info_by_index(d["hostApi"])["name"]
            devs.append((i, d["name"], api))
    return devs

def select_output_device(pa, prefer_name_substr="pulse", explicit_index=None):
    """
    Izberi prvo izhodno napravo, ki se ujema po imenu ali host API substringu (case-insensitive),
    s čimer elegantno obidemo ALSA in raje zadanejo PulseAudio/JACK/PipeWire preko PortAudio.
    """
    if explicit_index is not None:
        info = pa.get_device_info_by_index(int(explicit_index))
        if info.get("maxOutputChannels", 0) > 0:
            return int(explicit_index)
        raise RuntimeError(f"Naprava z indeksom {explicit_index} ni izhodna naprava.")

    s = (prefer_name_substr or "").lower()
    # Najprej poskusi po API-ju in/ali imenu (pulse, jack, pipewire, coreaudio, itd.)
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("maxOutputChannels", 0) <= 0:
            continue
        api = pa.get_host_api_info_by_index(d["hostApi"])["name"]
        name = d.get("name", "")
        if s and (s in name.lower() or s in api.lower()):
            return i

    # fallback: poskusi default output device (lahko je še vedno ALSA, ampak vsaj nekaj)
    try:
        return pa.get_default_output_device_info().get("index", None)
    except Exception:
        pass
    raise RuntimeError("Ni najdene izhodne naprave (PortAudio).")

# --------------------------- PortAudio init ---------------------------
p = pyaudio.PyAudio()

# informativno: izpiši razpoložljive izhodne naprave in API-je
for i, name, api in list_output_devices(p):
    print(f"[{i}] {name}  (host API: {api})")

try:
    out_index = select_output_device(p, prefer_name_substr=args.device_name,
                                     explicit_index=args.device_index)
    print(f"Uporabljam napravo index={out_index}")
    stream = p.open(format=pyaudio.paInt16,
                    channels=int(num_channels),
                    rate=int(sample_rate),
                    output=True,
                    output_device_index=out_index,
                    frames_per_buffer=int(samples_per_channel))
except Exception as e:
    p.terminate()
    sys.stderr.write(f"Napaka pri odprtju izhodne naprave (PortAudio): {e}\n")
    sys.exit(1)

# --------------------------- Branje iz FIFO in predvajanje ---------------------------
# Velikost bloka: 1 bajt (bit_width) + int8 vzorci (4096*2 kanala)
chunk_size = int(samples_per_channel * num_channels) + 1
print(f"Chunk Size (bytes): {chunk_size}")
print("Čakam na podatke v FIFO:", fifo_path)

# Če je fifo navadna datoteka, bo delalo; za pravi named pipe poskrbi z: mkfifo audiofifo1.fifo
if not os.path.exists(fifo_path):
    sys.stderr.write(f"Opozorilo: {fifo_path} ne obstaja. Ustvari z 'mkfifo {fifo_path}'.\n")

try:
    with open(fifo_path, 'rb') as fifo:
        while True:
            encoded_data = fifo.read(chunk_size)
            if not encoded_data:
                print("Ni več podatkov. Izhod.")
                break
            pcm_i16 = decode(encoded_data)
            # predvajaj prek PortAudio
            stream.write(pcm_i16.tobytes())

except KeyboardInterrupt:
    print("\nPrekinjeno z uporabniško kombinacijo (Ctrl+C).")
finally:
    # cleanup PortAudio
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()
