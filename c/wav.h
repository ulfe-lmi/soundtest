#ifndef WAV_H
#define WAV_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

int read_wav_header(FILE *f, int *sample_rate, int *channels, int *bits_per_sample, int *data_size);
FILE *open_wav_write(const char *filename, int sample_rate, int channels, int bits_per_sample);
void close_wav_write(FILE *f, int sample_rate, int channels, int bits_per_sample, int data_size);

#endif // WAV_H
