import numpy as np
import pyaudio
import sys

# Passthrough decode function
def decode(encoded_block):
    # No transformation, passthrough
    return encoded_block

# Main decoder program setup
fifo_path = "audiofifo1.fifo"

# Constants
sample_rate = 44100.0  # Sampling rate in Hz (float)
num_channels = 2.0     # Stereo (float)
samples_per_channel = 4096.0  # Fixed block size for each channel (float)

# Calculate chunk size and duration
chunk_size = int(samples_per_channel * num_channels * 2)  # Total bytes per block
chunk_duration = samples_per_channel / sample_rate  # Duration per block in seconds

print(f"Chunk Size: {chunk_size} bytes")
print(f"Chunk Duration: {chunk_duration:.6f} seconds")

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=int(num_channels), rate=int(sample_rate), output=True)

# Read from the FIFO and decode for playback
with open(fifo_path, 'rb') as fifo:
    while True:
        encoded_data = fifo.read(chunk_size)
        
        if not encoded_data:
            print("No data received, exiting decoder.")
            break

        # Pass data through without modification
        decoded_data = decode(encoded_data)

        # Play decoded data
        stream.write(decoded_data)

# Cleanup
stream.stop_stream()
stream.close()
p.terminate()
