import wave
import sys
import time

# Passthrough encode function
def encode(raw_data, bit_width=None):
    # No transformation, passthrough
    return raw_data

# Main encoder program setup
if len(sys.argv) < 2:
    print("Usage: python 00-encoder.py <input_wav_file>")
    sys.exit(1)

input_wav_file = sys.argv[1]
fifo_path = "audiofifo1.fifo"

sample_rate = 44100.0  # Sampling rate in Hz (float)
num_channels = 2.0     # Stereo (float)
samples_per_channel = 4096.0  # Fixed block size for each channel (float)

# Calculate chunk size and duration
chunk_size = int(samples_per_channel * num_channels * 2)  # Total bytes per block (2 bytes per 16-bit sample)
chunk_duration = samples_per_channel / sample_rate  # Duration per block in seconds

print(f"Chunk Size: {chunk_size} bytes")
print(f"Chunk Duration: {chunk_duration:.6f} seconds")

# Open the input WAV file
with wave.open(input_wav_file, 'rb') as wav:
    if wav.getframerate() != sample_rate or wav.getnchannels() != num_channels:
        print("Input file must be 44100 Hz, stereo.")
        sys.exit(1)
    TMAX = wav.getnframes() / sample_rate  # Total duration of the audio file in seconds

    with open(fifo_path, 'wb') as fifo:
        start_time = time.time()

        while True:
            frames = wav.readframes(int(samples_per_channel))
            if not frames:
                break

            # Pass data through without modification
            encoded_data = encode(frames)

            # Write raw data to FIFO
            fifo.write(encoded_data)

            # Sleep for the duration of the chunk
            time.sleep(chunk_duration - (time.time() - start_time) % chunk_duration)
