#!/usr/bin/env python3
import argparse
import numpy as np
import pyaudio
import wave
import sys
from scipy.fftpack import idct

# Global variables to store the last sample of each channel from the previous block
last_sample_left = 0
last_sample_right = 0

# (podvojeni importi iz izvirnika puščam pri miru)
import numpy as np
from scipy.fftpack import idct
import sys

# Global variables to store the last sample for continuity correction
last_sample_left = 0
last_sample_right = 0

# --------------------------- Argparser (DODANO: izbira PortAudio naprave) ---------------------------
ap = argparse.ArgumentParser(description="DCT 8-bit finetune decoder with selectable PortAudio output device.")
ap.add_argument("--fifo", default="audiofifo1.fifo", help="Path do named pipe/FIFO.")
ap.add_argument("--device-name", default="pulse",
                help="Niz, ki se ujema z imenom naprave ali host API (npr. 'pulse', 'pipewire', 'jack').")
ap.add_argument("--device-index", type=int, default=None,
                help="Eksplicitni PyAudio index izhodne naprave (prepiše --device-name).")
ap.add_argument("--sample-rate", type=float, default=44100.0, help="Vz. frekvenca (Hz).")
ap.add_argument("--channels", type=float, default=2.0, help="Število kanalov.")
ap.add_argument("--block", type=float, default=4096.0, help="Vzorcev/kanal na blok.")
ap.add_argument("--debug-wav", default="debug.wav", help="Izhodna WAV datoteka za snemanje dekodiranega zvoka.")
args = ap.parse_args()

# --------------------------- Decode (ne spreminjam) ---------------------------
def decode(encoded_block):
    global last_sample_left, last_sample_right
    
    # Read the first byte to determine max_value
    max_value = encoded_block[0]
    print(f"Decoder: Retrieved max_value = {max_value}")

    # Read the next 8 bytes to get the max_coeff for scaling
    max_coeff = np.frombuffer(encoded_block[1:9], dtype=np.float64)[0]
    print(f"Decoder: Retrieved max_coeff = {max_coeff}")

    # Check if max_coeff or max_value is valid before proceeding
    if max_value == 0 or max_coeff == 0:
        print("Warning: max_value or max_coeff is zero, skipping this block.")
        return np.zeros(int(samples_per_channel * num_channels), dtype=np.int16)

    # Convert the rest of the block to numpy array for de-quantization
    quantized_coeffs = np.frombuffer(encoded_block[9:], dtype=np.int8).reshape(-1, 2)

    # De-quantize the DCT coefficients
    dct_coeffs = (quantized_coeffs.astype(np.float32) / max_value) * max_coeff

    # Perform IDCT to reconstruct the audio data as float
    decoded_data = idct(dct_coeffs, type=2, norm='ortho', axis=0).astype(np.float32)

    # Check for maximum absolute value and normalize if necessary
    max_abs_value = np.max(np.abs(decoded_data))
    int16_max_value = np.iinfo(np.int16).max  # 32767

    if max_abs_value > int16_max_value:
        # Scale down to prevent clipping
        decoded_data = (decoded_data / max_abs_value) * int16_max_value        

    # additional protection against clipping
    decoded_data = decoded_data * 0.95

    # Convert to int16 for playback
    decoded_data = decoded_data.astype(np.int16)

    # Ramp correction for block continuity
    ramp_length = int(0.2 * len(decoded_data))  # 20% of the block length (opomba: originalni komentar je 10%)

    # Apply ramp to both channels separately
    left_channel = decoded_data[:, 0]
    right_channel = decoded_data[:, 1]

    # Create ramped start for each channel
    if ramp_length > 0:
        left_ramp = np.linspace(last_sample_left, left_channel[0], ramp_length)
        right_ramp = np.linspace(last_sample_right, right_channel[0], ramp_length)

        left_channel[:ramp_length] = left_ramp + left_channel[:ramp_length] - left_channel[0]
        right_channel[:ramp_length] = right_ramp + right_channel[:ramp_length] - right_channel[0]

    # Update the last sample with the end of the current block for each channel
    last_sample_left = left_channel[-1]
    last_sample_right = right_channel[-1]

    # Flatten the stereo channels back for playback
    decoded_data[:, 0] = left_channel
    decoded_data[:, 1] = right_channel
    return decoded_data.flatten()

# --------------------------- Parametri iz argov (ohranim izvirne tipe kot v kodi) ---------------------------
fifo_path = args.fifo
debug_wav_file = args.debug_wav
sample_rate = args.sample_rate      # float (v izvirniku je float)
num_channels = args.channels        # float
samples_per_channel = args.block    # float

# Calculate chunk size and duration (ne spreminjam izraza)
chunk_size = int(samples_per_channel * num_channels) + 1 + 8  # audio vzorci + 1B header + 8B max_coeff
chunk_duration = samples_per_channel / sample_rate  # Duration per block in seconds

print(f"Chunk Size: {chunk_size} bytes")
print(f"Chunk Duration: {chunk_duration:.6f} seconds")

# --------------------------- PortAudio helperji (DODANO) ---------------------------
def list_output_devices(pa):
    devs = []
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("maxOutputChannels", 0) > 0:
            api = pa.get_host_api_info_by_index(d["hostApi"])["name"]
            devs.append((i, d["name"], api))
    return devs

def select_output_device(pa, prefer_name_substr="pulse", explicit_index=None):
    if explicit_index is not None:
        info = pa.get_device_info_by_index(int(explicit_index))
        if info.get("maxOutputChannels", 0) > 0:
            return int(explicit_index)
        raise RuntimeError(f"Naprava z indeksom {explicit_index} ni izhodna naprava.")

    s = (prefer_name_substr or "").lower()
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if d.get("maxOutputChannels", 0) <= 0:
            continue
        api = pa.get_host_api_info_by_index(d["hostApi"])["name"]
        name = d.get("name", "")
        if s and (s in name.lower() or s in api.lower()):
            return i

    # fallback na privzeto
    try:
        return pa.get_default_output_device_info().get("index", None)
    except Exception:
        pass
    raise RuntimeError("Ni najdene izhodne naprave (PortAudio).")

# --------------------------- Inicializacija PortAudio + WAV zapis (sprememba samo pri out napravi) ---------------------------
p = pyaudio.PyAudio()

# informativni izpis naprav (pomaga izbrati backend brez ALSA)
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

wav_output = wave.open(debug_wav_file, 'wb')
wav_output.setnchannels(int(num_channels))
wav_output.setsampwidth(2)  # 2 bytes za 16-bit audio
wav_output.setframerate(int(sample_rate))

# --------------------------- Zanka: branje FIFO -> decode -> playback + zapis ---------------------------
try:
    with open(fifo_path, 'rb') as fifo:
        while True:
            encoded_data = fifo.read(chunk_size)
            if not encoded_data:
                print("No data received, exiting decoder.")
                break

            # Decode one block of data (ne spreminjam)
            decoded_data = decode(encoded_data)

            # Play and save decoded data
            stream.write(decoded_data.tobytes())
            wav_output.writeframes(decoded_data.tobytes())

except KeyboardInterrupt:
    print("Decoder interrupted by user.")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Cleanup resources
    try:
        stream.stop_stream()
        stream.close()
    except Exception:
        pass
    p.terminate()
    wav_output.close()
    print(f"Debug WAV file saved as {debug_wav_file}.")
