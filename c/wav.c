#include "wav.h"
#include <string.h>

int read_wav_header(FILE *f, int *sample_rate, int *channels, int *bits_per_sample, int *data_size) {
    unsigned char header[44];
    if (fread(header, 1, 44, f) != 44) return -1;
    if (memcmp(header, "RIFF", 4) != 0) return -1;
    if (memcmp(header + 8, "WAVE", 4) != 0) return -1;
    if (memcmp(header + 12, "fmt ", 4) != 0) return -1;
    int fmt_size = header[16] | (header[17] << 8) | (header[18] << 16) | (header[19] << 24);
    if (fmt_size < 16) return -1;
    int audio_format = header[20] | (header[21] << 8);
    *channels = header[22] | (header[23] << 8);
    *sample_rate = header[24] | (header[25] << 8) | (header[26] << 16) | (header[27] << 24);
    *bits_per_sample = header[34] | (header[35] << 8);
    if (audio_format != 1) return -1; // PCM only
    // Find 'data' chunk
    long pos = 12 + 8 + fmt_size;
    fseek(f, pos, SEEK_SET);
    while (1) {
        unsigned char chunk[8];
        if (fread(chunk, 1, 8, f) != 8) return -1;
        int size = chunk[4] | (chunk[5] << 8) | (chunk[6] << 16) | (chunk[7] << 24);
        if (memcmp(chunk, "data", 4) == 0) {
            *data_size = size;
            return 0;
        }
        fseek(f, size, SEEK_CUR);
    }
}

FILE *open_wav_write(const char *filename, int sample_rate, int channels, int bits_per_sample) {
    FILE *f = fopen(filename, "wb");
    if (!f) return NULL;
    unsigned char header[44] = {0};
    memcpy(header, "RIFF", 4);
    memcpy(header + 8, "WAVE", 4);
    memcpy(header + 12, "fmt ", 4);
    int fmt_size = 16;
    header[16] = fmt_size & 0xFF;
    header[17] = (fmt_size >> 8) & 0xFF;
    header[18] = (fmt_size >> 16) & 0xFF;
    header[19] = (fmt_size >> 24) & 0xFF;
    header[20] = 1; // PCM
    header[22] = channels & 0xFF;
    header[23] = (channels >> 8) & 0xFF;
    header[24] = sample_rate & 0xFF;
    header[25] = (sample_rate >> 8) & 0xFF;
    header[26] = (sample_rate >> 16) & 0xFF;
    header[27] = (sample_rate >> 24) & 0xFF;
    int byte_rate = sample_rate * channels * bits_per_sample / 8;
    header[28] = byte_rate & 0xFF;
    header[29] = (byte_rate >> 8) & 0xFF;
    header[30] = (byte_rate >> 16) & 0xFF;
    header[31] = (byte_rate >> 24) & 0xFF;
    int block_align = channels * bits_per_sample / 8;
    header[32] = block_align & 0xFF;
    header[33] = (block_align >> 8) & 0xFF;
    header[34] = bits_per_sample & 0xFF;
    header[35] = (bits_per_sample >> 8) & 0xFF;
    memcpy(header + 36, "data", 4);
    fwrite(header, 1, 44, f);
    return f;
}

void close_wav_write(FILE *f, int sample_rate, int channels, int bits_per_sample, int data_size) {
    int file_size = 36 + data_size;
    int byte_rate = sample_rate * channels * bits_per_sample / 8;
    int block_align = channels * bits_per_sample / 8;
    unsigned char header[44];
    fseek(f, 0, SEEK_SET);
    memcpy(header, "RIFF", 4);
    header[4] = file_size & 0xFF;
    header[5] = (file_size >> 8) & 0xFF;
    header[6] = (file_size >> 16) & 0xFF;
    header[7] = (file_size >> 24) & 0xFF;
    memcpy(header + 8, "WAVE", 4);
    memcpy(header + 12, "fmt ", 4);
    header[16] = 16; header[17]=0; header[18]=0; header[19]=0;
    header[20] = 1; header[21] = 0;
    header[22] = channels & 0xFF; header[23] = (channels >> 8) & 0xFF;
    header[24] = sample_rate & 0xFF;
    header[25] = (sample_rate >> 8) & 0xFF;
    header[26] = (sample_rate >> 16) & 0xFF;
    header[27] = (sample_rate >> 24) & 0xFF;
    header[28] = byte_rate & 0xFF;
    header[29] = (byte_rate >> 8) & 0xFF;
    header[30] = (byte_rate >> 16) & 0xFF;
    header[31] = (byte_rate >> 24) & 0xFF;
    header[32] = block_align & 0xFF;
    header[33] = (block_align >> 8) & 0xFF;
    header[34] = bits_per_sample & 0xFF;
    header[35] = (bits_per_sample >> 8) & 0xFF;
    memcpy(header + 36, "data", 4);
    header[40] = data_size & 0xFF;
    header[41] = (data_size >> 8) & 0xFF;
    header[42] = (data_size >> 16) & 0xFF;
    header[43] = (data_size >> 24) & 0xFF;
    fwrite(header, 1, 44, f);
    fclose(f);
}
