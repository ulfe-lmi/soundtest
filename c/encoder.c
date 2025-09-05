#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "dct.h"
#include "entropy.h"
#include "wav.h"

#define BLOCK_SIZE 4096
#define CHANNELS 2

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_wav_file> [target_entropy]\n", argv[0]);
        return 1;
    }
    const char *input_wav_file = argv[1];
    double target_entropy = (argc > 2) ? atof(argv[2]) : 2.0;
    const char *fifo_path = "audiofifo1.fifo";

    FILE *wav = fopen(input_wav_file, "rb");
    if (!wav) { perror("open input wav"); return 1; }
    int sample_rate, channels, bits_per_sample, data_size;
    if (read_wav_header(wav, &sample_rate, &channels, &bits_per_sample, &data_size) != 0 ||
        sample_rate != 44100 || channels != 2 || bits_per_sample != 16) {
        fprintf(stderr, "Input file must be 44100 Hz, stereo, 16-bit PCM.\n");
        fclose(wav);
        return 1;
    }

    FILE *fifo = fopen(fifo_path, "wb");
    if (!fifo) { perror("open fifo"); fclose(wav); return 1; }

    int16_t buffer[BLOCK_SIZE * CHANNELS];
    double chan[CHANNELS][BLOCK_SIZE];
    double dct_coeffs[CHANNELS][BLOCK_SIZE];
    int8_t best_quantized[BLOCK_SIZE * CHANNELS];
    int8_t temp_quantized[BLOCK_SIZE * CHANNELS];

    while (1) {
        size_t read = fread(buffer, sizeof(int16_t) * CHANNELS, BLOCK_SIZE, wav);
        if (read == 0) break;
        for (size_t i = 0; i < read; ++i) {
            chan[0][i] = buffer[2 * i];
            chan[1][i] = buffer[2 * i + 1];
        }
        if (read < BLOCK_SIZE) {
            for (size_t i = read; i < BLOCK_SIZE; ++i) {
                chan[0][i] = 0;
                chan[1][i] = 0;
            }
        }

        dct_ii(chan[0], dct_coeffs[0], BLOCK_SIZE);
        dct_ii(chan[1], dct_coeffs[1], BLOCK_SIZE);

        double closest = 1e9;
        int best_max_value = 1;
        double best_max_coeff = 0;
        for (int max_value = 1; max_value <= 255; max_value += 4) {
            double max_coeff = fabs(dct_coeffs[0][0]);
            for (int ch = 0; ch < CHANNELS; ++ch) {
                for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                    double val = fabs(dct_coeffs[ch][i]);
                    if (val > max_coeff) max_coeff = val;
                }
            }
            if (max_coeff == 0) max_coeff = 1;
            for (int ch = 0; ch < CHANNELS; ++ch) {
                for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                    double norm = dct_coeffs[ch][i] / max_coeff;
                    int8_t q = (int8_t)(norm * max_value);
                    temp_quantized[i * CHANNELS + ch] = q;
                }
            }
            double ent = calculate_byte_entropy(temp_quantized, BLOCK_SIZE * CHANNELS);
            double diff = fabs(ent - target_entropy);
            if (diff < closest) {
                closest = diff;
                best_max_value = max_value;
                best_max_coeff = max_coeff;
                memcpy(best_quantized, temp_quantized, BLOCK_SIZE * CHANNELS);
            }
        }

        fprintf(stdout, "Selected max value: %d, Achieved Entropy: %.2f\n",
                best_max_value, calculate_byte_entropy(best_quantized, BLOCK_SIZE * CHANNELS));
        double in_entropy = calculate_16bit_entropy(buffer, read * CHANNELS);
        double out_entropy = calculate_byte_entropy(best_quantized, BLOCK_SIZE * CHANNELS);
        fprintf(stdout, "InEntropy: %.2f bits  OutEntropy: %.2f bits\n", in_entropy, out_entropy);

        fputc(best_max_value, fifo);
        fwrite(&best_max_coeff, sizeof(double), 1, fifo);
        fwrite(best_quantized, 1, BLOCK_SIZE * CHANNELS, fifo);
    }

    fclose(fifo);
    fclose(wav);
    return 0;
}
