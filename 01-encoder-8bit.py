import wave
import numpy as np
import sys
import time
from entropy_calculator import calculate_byte_entropy
from entropy_calculator import calculate_16bit_entropy
from gauge_display import GaugeDisplay

def encode(raw_data, bit_width):
    # Convert 16-bit raw data to numpy array
    audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
    # Scale down to the desired bit width
    audio_data_scaled = (audio_data / (2 ** (16 - bit_width))).astype(np.int8).flatten()
    
    # Add bit_width as the first int8 byte of the encoded block
    encoded_block = bytearray(np.array([bit_width], dtype=np.int8).tobytes()) + audio_data_scaled.tobytes()
 
    return encoded_block

# Main encoder program setup
if len(sys.argv) < 2:
    print("Usage: python 01-encoder-8bit.py <input_wav_file> [bit_width]")
    sys.exit(1)

input_wav_file = sys.argv[1]
bit_width = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # Default to 8 if not provided
fifo_path = "audiofifo1.fifo"

sample_rate = 44100.0  # Sampling rate in Hz (float)
num_channels = 2.0     # Stereo (float)
samples_per_channel = 4096.0  # Fixed block size for each channel (float)

# Calculate chunk size and duration
chunk_size = samples_per_channel * num_channels  # Total samples per block for stereo
chunk_duration = chunk_size / (sample_rate * num_channels)  # Duration per block in seconds

print(f"Chunk Size: {chunk_size} samples")
print(f"Chunk Duration: {chunk_duration:.6f} seconds")

initial_buffer_chunks = 2

# Initialize GaugeDisplay
#gauge = GaugeDisplay()
#gauge.gauge_initialize("INPUT ENTROPY:", "OUTPUT ENTROPY")

# Open the input WAV file
with wave.open(input_wav_file, 'rb') as wav:
    if wav.getframerate() != sample_rate or wav.getnchannels() != num_channels:
        print("Input file must be 44100 Hz, stereo.")
        sys.exit(1)
    TMAX = wav.getnframes() / sample_rate  # Total duration of the audio file in seconds

    with open(fifo_path, 'wb') as fifo:
        chunk_count = 0
        start_time = time.time()

        while True:
            loop_start_time = time.time()
            frames = wav.readframes(int(samples_per_channel))
            if not frames:
                break

            # Encode one block of data
            encoded_data = encode(frames, bit_width)

            # Calculate entropy and update gauge
            in_entropy = calculate_16bit_entropy(frames)
            out_entropy = calculate_byte_entropy(encoded_data)
            elapsed_time = time.time() - start_time
            time_percentage = min((elapsed_time / TMAX) * 100, 100)  # Capped at 100%

            # Update the gauge display in percentages
 #           gauge.update_display(in_entropy / 8 * 100, out_entropy / 8 * 100, time_percentage, elapsed_time)

            # Print entropy and bitrate information to the console
            print(f"InEntropy: {in_entropy:.2f} bits  OutEntropy: {out_entropy:.2f} bits")

            # Write encoded data to FIFO
            fifo.write(encoded_data)

            # Initial buffering and timing adjustments
            elapsed_iteration_time = time.time() - loop_start_time
            if chunk_count < 2:
                chunk_count += 1
            else:
                sleep_time = max(0, chunk_duration - elapsed_iteration_time)
                time.sleep(sleep_time)

# Clean up gauge display
#gauge.gauge_delete()
