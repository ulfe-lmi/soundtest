# entropy_calculator.py

import math
from collections import Counter

def calculate_byte_entropy(data):
    """
    Calculate the entropy of a byte buffer based on 8-bit symbols.
    
    Parameters:
        data (bytes or bytearray): The byte buffer containing data.

    Returns:
        float: The calculated entropy in bits per byte.
    """
    byte_counts = Counter(data)
    total_bytes = len(data)
    entropy = 0.0
    for count in byte_counts.values():
        p_x = count / total_bytes
        entropy -= p_x * math.log2(p_x)
    return entropy

def calculate_16bit_entropy(data):
    """
    Calculate the entropy of a buffer of 16-bit values from audio data.

    Parameters:
        data (bytes or bytearray): The byte buffer containing 16-bit values,
                                   as in frames from a 16-bit WAV file.
    
    Returns:
        float: The calculated entropy in bits per 16-bit symbol.
    
    Raises:
        ValueError: If the length of data is not even, as 16-bit values require pairs of bytes.
    """
    if len(data) % 2 != 0:
        raise ValueError("Data length must be even to represent 16-bit values.")

    # Convert the byte buffer to 16-bit signed integers
    values = [int.from_bytes(data[i:i+2], byteorder='little', signed=True) for i in range(0, len(data), 2)]
    
    # Count occurrences of each 16-bit value
    value_counts = Counter(values)
    total_values = len(values)
    
    # Calculate the entropy
    entropy = 0.0
    for count in value_counts.values():
        p_x = count / total_values
        entropy -= p_x * math.log2(p_x)
    
    return entropy

# Optional: Test the functions when running this file directly
if __name__ == "__main__":
    import wave

    # Example usage of 8-bit entropy calculation
    sample_data = b"example data for entropy calculation"
    entropy_byte = calculate_byte_entropy(sample_data)
    print("Byte-level Entropy:", entropy_byte)

    # Example 16-bit entropy calculation using a WAV file
    try:
        with wave.open("example_16bit_audio.wav", "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            entropy_16bit = calculate_16bit_entropy(frames)
            print("16-bit Audio Entropy:", entropy_16bit)
    except FileNotFoundError:
        print("Example WAV file not found.")
