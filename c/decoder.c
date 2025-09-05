#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "dct.h"
#include "wav.h"

#define BLOCK_SIZE 4096
#define CHANNELS 2

int main(int argc, char *argv[]) {
    const char *fifo_path = "audiofifo1.fifo";
    const char *output_wav = (argc > 1) ? argv[1] : "debug.wav";

    FILE *fifo = fopen(fifo_path, "rb");
    if (!fifo) { perror("open fifo"); return 1; }

    FILE *out = open_wav_write(output_wav, 44100, 2, 16);
    if (!out) { perror("open output wav"); fclose(fifo); return 1; }

    double dct_coeffs[CHANNELS][BLOCK_SIZE];
    double samples[CHANNELS][BLOCK_SIZE];
    int16_t output_buffer[BLOCK_SIZE * CHANNELS];
    double last_left = 0.0, last_right = 0.0;
    int data_written = 0;

    while (1) {
        int max_value = fgetc(fifo);
        if (max_value == EOF) break;
        double max_coeff;
        if (fread(&max_coeff, sizeof(double), 1, fifo) != 1) break;
        int8_t quantized[BLOCK_SIZE * CHANNELS];
        if (fread(quantized, 1, BLOCK_SIZE * CHANNELS, fifo) != BLOCK_SIZE * CHANNELS) break;

        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            for (int ch = 0; ch < CHANNELS; ++ch) {
                double norm = (double)quantized[i * CHANNELS + ch] / (double)max_value;
                dct_coeffs[ch][i] = norm * max_coeff;
            }
        }

        idct_ii(dct_coeffs[0], samples[0], BLOCK_SIZE);
        idct_ii(dct_coeffs[1], samples[1], BLOCK_SIZE);

        double max_abs = 0.0;
        for (int ch = 0; ch < CHANNELS; ++ch) {
            for (size_t i = 0; i < BLOCK_SIZE; ++i) {
                if (fabs(samples[ch][i]) > max_abs) max_abs = fabs(samples[ch][i]);
            }
        }
        if (max_abs > 32767.0) {
            double scale = 32767.0 / max_abs;
            for (int ch = 0; ch < CHANNELS; ++ch)
                for (size_t i = 0; i < BLOCK_SIZE; ++i)
                    samples[ch][i] *= scale;
        }
        for (int ch = 0; ch < CHANNELS; ++ch)
            for (size_t i = 0; i < BLOCK_SIZE; ++i)
                samples[ch][i] *= 0.95;

        int ramp_length = (int)(0.2 * BLOCK_SIZE);
        if (ramp_length > 0) {
            for (int i = 0; i < ramp_length; ++i) {
                double t = (double)i / ramp_length;
                double left_ramp = last_left + (samples[0][0] - last_left) * t;
                double right_ramp = last_right + (samples[1][0] - last_right) * t;
                samples[0][i] = left_ramp + samples[0][i] - samples[0][0];
                samples[1][i] = right_ramp + samples[1][i] - samples[1][0];
            }
        }
        last_left = samples[0][BLOCK_SIZE - 1];
        last_right = samples[1][BLOCK_SIZE - 1];

        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            int16_t l = (int16_t)lrint(samples[0][i]);
            int16_t r = (int16_t)lrint(samples[1][i]);
            output_buffer[2 * i] = l;
            output_buffer[2 * i + 1] = r;
        }
        fwrite(output_buffer, sizeof(int16_t) * CHANNELS, BLOCK_SIZE, out);
        data_written += BLOCK_SIZE * CHANNELS * sizeof(int16_t);
    }

    close_wav_write(out, 44100, 2, 16, data_written);
    fclose(fifo);
    return 0;
}
