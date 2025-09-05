import wave
import numpy as np
import sys
import time
from entropy_calculator import calculate_byte_entropy
from entropy_calculator import calculate_16bit_entropy
from gauge_display import GaugeDisplay
from scipy.fftpack import dct, idct


def apply_nms_on_dct(dct_coeffs, window_length, ratio_threshold):
    """
    Applies non-maximum suppression (NMS) to DCT coefficients.
    Only the coefficient with the maximum absolute value is retained in each window, 
    and only coefficients above the ratio threshold relative to the maximum are retained.
    
    Parameters:
        dct_coeffs (np.ndarray): The DCT coefficients array (2D for stereo channels).
        window_length (int): The length of the window for NMS.
        ratio_threshold (float): Threshold ratio (between 0 and 1) relative to the maximum value.
        
    Returns:
        np.ndarray: The modified DCT coefficients after applying NMS with threshold.
    """
    # Ensure we're working with a copy to avoid modifying the original coefficients
    dct_coeffs_nms = np.copy(dct_coeffs)

    # Apply NMS on each channel separately
    for channel in range(dct_coeffs.shape[1]):
        # Slide the window across the coefficients
        for i in range(0, dct_coeffs_nms.shape[0], window_length):
            # Define the current window range
            window_end = min(i + window_length, dct_coeffs_nms.shape[0])
            window = dct_coeffs_nms[i:window_end, channel]

            # Find the maximum absolute value in this window
            if len(window) > 0:
                max_idx = np.argmax(np.abs(window))
                max_value = window[max_idx]

                # Zero out all coefficients below the threshold ratio
                for j in range(len(window)):
                    if abs(window[j]) < abs(max_value) * ratio_threshold:
                        window[j] = 0

                # Only keep the maximum value in the window
                dct_coeffs_nms[i:window_end, channel] = window
                dct_coeffs_nms[i + max_idx, channel] = max_value
    return dct_coeffs_nms



def encode(raw_data, target_entropy):
    closest_entropy_diff = float('inf')
    best_max_value = 1
    best_encoded_data = None
    best_max_coeff = 0

    # Convert 16-bit raw data to numpy array
    audio_data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
    
    # Perform DCT on each channel separately
    dct_coeffs = dct(audio_data, type=2, norm='ortho', axis=0)
    
    window_length = 32  # Define the NMS window length
    ratio_threshold = 0.3
    dct_coeffs = apply_nms_on_dct(dct_coeffs, window_length, ratio_threshold)

    
    # Try encoding with each max_value from 1 to 255
    for max_value in range(1, 256,4):
        # Calculate max coefficient for normalization
        max_coeff = np.max(np.abs(dct_coeffs))
        
        # Normalize DCT coefficients to the range [-1, 1]
        if max_coeff > 0:
            dct_normalized = dct_coeffs / max_coeff
        else:
            dct_normalized = dct_coeffs  # If max_coeff is 0, skip normalization

        # Quantize normalized DCT coefficients to the current max_value
        dct_quantized = (dct_normalized * max_value).astype(np.int8).flatten()

        # Calculate entropy for the current encoding
        entropy = calculate_byte_entropy(dct_quantized.tobytes())
        
        # Print max_value and entropy for this iteration
        #print(f"Max value: {max_value}, Entropy: {entropy:.2f}")

        # Check if this encoding's entropy is closer to the target_entropy
        entropy_diff = abs(entropy - target_entropy)
        if entropy_diff < closest_entropy_diff:
            closest_entropy_diff = entropy_diff
            best_max_value = max_value
            best_encoded_data = dct_quantized
            best_max_coeff = max_coeff  # Store the max_coeff associated with this max_value

    # Print final selected max_value and entropy
    print(f"Selected max value: {best_max_value}, Achieved Entropy: {calculate_byte_entropy(best_encoded_data.tobytes()):.2f}")

    # Add selected max_value and max_coeff as metadata for decoding
    encoded_block = bytearray([best_max_value]) + bytearray(np.float64(best_max_coeff).tobytes()) + best_encoded_data.tobytes()
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
