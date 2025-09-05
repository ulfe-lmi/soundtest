import wave
import numpy as np
import sys
import time
from entropy_calculator import calculate_byte_entropy
from entropy_calculator import calculate_16bit_entropy
from gauge_display import GaugeDisplay
from scipy.fftpack import dct, idct

def encode(raw_data, target_entropy):
    closest_entropy_diff = float('inf')
    best_bit_width = 1
    best_encoded_data = None
    best_max_coeff = 0

    # Convert 16-bit raw data to numpy array
    audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
    
    # Perform DCT on each channel separately
    dct_coeffs = dct(audio_data, type=2, norm='ortho', axis=0)
    
    # Try encoding with each bit width from 1 to 8
    for bit_width in range(1, 9):
        max_value = 2 ** (bit_width - 1) - 1

        # Calculate max coefficient for normalization
        max_coeff = np.max(np.abs(dct_coeffs))
        
        # Normalize DCT coefficients to the range [-1, 1]
        if max_coeff > 0:
            dct_normalized = dct_coeffs / max_coeff
        else:
            dct_normalized = dct_coeffs  # If max_coeff is 0, skip normalization

        # Quantize normalized DCT coefficients to the current bit width
        dct_quantized = (dct_normalized * max_value).astype(np.int8).flatten()

        # Calculate entropy for the current encoding
        entropy = calculate_byte_entropy(dct_quantized.tobytes())
        
        # Print bit width and entropy for this iteration
        print(f"Bit width: {bit_width}, Entropy: {entropy:.2f}")

        # Check if this encoding's entropy is closer to the target_entropy
        entropy_diff = abs(entropy - target_entropy)
        if entropy_diff < closest_entropy_diff:
            closest_entropy_diff = entropy_diff
            best_bit_width = bit_width
            best_encoded_data = dct_quantized
            best_max_coeff = max_coeff  # Store the max_coeff associated with this bit width

    # Print final selected bit width and entropy
    print(f"Selected bit width: {best_bit_width}, Achieved Entropy: {calculate_byte_entropy(best_encoded_data.tobytes()):.2f}")

    # Add selected bit width and max_coeff as metadata for decoding
    encoded_block = bytearray([best_bit_width]) + bytearray(np.float64(best_max_coeff).tobytes()) + best_encoded_data.tobytes()
    return encoded_block

# Main encoder program setup
if len(sys.argv) < 2:
    print("Usage: python 01-encoder-8bit.py <input_wav_file> [target_entropy]")
    sys.exit(1)

input_wav_file = sys.argv[1]
target_entropy = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0  # Default to 2.0 bits if not provided
fifo_path = "audiofifo1.fifo"

# Updated fixed block length of 4096 samples per channel
sample_rate = 44100.0  # Sampling rate in Hz (float)
num_channels = 2.0     # Stereo (float)
samples_per_channel = 4096.0  # Fixed block size for each channel (float)

# Calculate chunk size and duration based on block length
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
            encoded_data = encode(frames, target_entropy)

            # Calculate entropy and update gauge
            in_entropy = calculate_16bit_entropy(frames)
            out_entropy = calculate_byte_entropy(encoded_data)
            elapsed_time = time.time() - start_time
            time_percentage = min((elapsed_time / TMAX) * 100, 100)  # Capped at 100%

            # Update the gauge display in percentages
            #gauge.update_display(in_entropy / 8 * 100, out_entropy / 8 * 100, time_percentage, elapsed_time)

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
