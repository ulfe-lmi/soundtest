import numpy as np
import pyaudio
import os
import sys

def decode(encoded_block):
    # Read the first byte as bit_width
    bit_width = encoded_block[0]
    print(f"Decoding with bit width: {bit_width}")

    # Interpret only the audio data (excluding the first byte) as int8 and reshape
    audio_data = np.frombuffer(encoded_block[1:], dtype=np.int8).reshape(-1, 2)

    # Scale up to 16-bit range based on the bit width
    scaling_factor = 2 ** (16 - bit_width)
    decoded_data = (audio_data.astype(np.int16) * scaling_factor).flatten()
    return decoded_data

# Main decoder program setup
fifo_path = "audiofifo1.fifo"

# Constants
sample_rate = 44100.0  # Sampling rate in Hz (float)
num_channels = 2.0     # Stereo (float)
samples_per_channel = 4096.0  # Fixed block size for each channel (float)
bit_width = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # Default bit width is 8

# Calculate chunk size and duration
chunk_size = samples_per_channel * num_channels  # Total samples per block plus 1 byte for bit_width
chunk_duration = (chunk_size) / (sample_rate * num_channels)  # Duration per block in seconds

print(f"Chunk Size: {chunk_size} samples")
print(f"Chunk Duration: {chunk_duration:.6f} seconds")


# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=int(num_channels), rate=int(sample_rate), output=True)

# Read from the FIFO and decode for playback
with open(fifo_path, 'rb') as fifo:
    while True:
        encoded_data = fifo.read(int(chunk_size)+1)
        
        if not encoded_data:
            print("No data received, exiting decoder.")
            break

        # Decode one block of data
        decoded_data = decode(encoded_data)
        # Play decoded data
        stream.write(decoded_data.tobytes())

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
