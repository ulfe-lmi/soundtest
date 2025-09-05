# C Implementation of 8-bit DCT Finetune Encoder/Decoder

This directory contains a pure C reimplementation of the Python tools
`03-encoder-8bit-dct-finetune.py` and `03-decoder-8bit-dct-finetune.py`.
The goal is to provide a standalone project that can be built with a
simple `make` command and run without any Python dependencies.

## Features

* Processes stereo WAV files (44.1 kHz, 16-bit PCM).
* Encodes audio in blocks of 4096 samples per channel using a
  Discrete Cosine Transform (DCT-II) with orthonormal scaling.
* Searches for an 8-bit quantization level that matches a target
  Shannon entropy.
* Decoder reconstructs audio using the inverse DCT and writes the
  result to a WAV file while applying a small ramp to reduce block
  discontinuities.
* Data is transferred between encoder and decoder through a UNIX
  named pipe (`audiofifo1.fifo`) in the same format as the original
  Python scripts.

## Project Layout

```
Makefile       – build rules for `encoder` and `decoder`
dct.c/.h       – orthonormal DCT-II and inverse DCT implementations
entropy.c/.h   – entropy calculation helpers
wav.c/.h       – minimal WAV reader/writer for 16-bit PCM
encoder.c      – command line encoder
decoder.c      – command line decoder
```

## Building

This project uses only the C standard library and `gcc`. On Ubuntu
install the build tools with:

```bash
sudo apt-get install build-essential
```

Compile the project:

```bash
cd c
make
```

This produces two executables: `encoder` and `decoder`.

## Running

1. **Prepare a FIFO**

   ```bash
   mkfifo audiofifo1.fifo
   ```

2. **Run the encoder**

   ```bash
   ./encoder input.wav 2.0
   ```

   The encoder reads `input.wav`, searches for a quantization level
   that achieves the target entropy (2.0 bits per byte by default) and
   writes encoded blocks to `audiofifo1.fifo`.

3. **Run the decoder**

   In another terminal start the decoder which reads from the FIFO and
   writes a reconstructed WAV file (`debug.wav` by default):

   ```bash
   ./decoder output.wav
   ```

   After the encoder finishes, the decoded audio will be available in
   `output.wav`.

4. **Play the result**

   You can listen to the output using any WAV player, e.g. `aplay`:

   ```bash
   aplay output.wav
   ```

## Notes

* The DCT and entropy calculations are implemented directly in C to
  avoid external dependencies. While this is sufficient for small
  experiments, it is slower than optimized libraries such as FFTW.
* The implementation mirrors the structure of the original Python
  scripts but omits real-time audio playback. The decoder instead
  writes the reconstructed signal to a WAV file.
* Block size and other parameters can be adjusted in the source code
  if desired.

## License

This code is provided under the same license as the rest of the
repository. See `../LICENSE` for details.
