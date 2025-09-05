#include "entropy.h"
#include <math.h>

#define BYTE_VALUES 256

double calculate_byte_entropy(const int8_t *data, size_t length) {
    if (length == 0) return 0.0;
    unsigned int counts[BYTE_VALUES] = {0};
    for (size_t i = 0; i < length; ++i) {
        unsigned char b = (unsigned char)data[i];
        counts[b]++;
    }
    double entropy = 0.0;
    for (int i = 0; i < BYTE_VALUES; ++i) {
        if (counts[i] != 0) {
            double p = (double)counts[i] / (double)length;
            entropy -= p * (log(p) / log(2.0));
        }
    }
    return entropy;
}

double calculate_16bit_entropy(const int16_t *data, size_t length) {
    if (length == 0) return 0.0;
    unsigned int counts[65536] = {0};
    for (size_t i = 0; i < length; ++i) {
        unsigned short v = (unsigned short)(data[i] + 32768);
        counts[v]++;
    }
    double entropy = 0.0;
    for (int i = 0; i < 65536; ++i) {
        if (counts[i] != 0) {
            double p = (double)counts[i] / (double)length;
            entropy -= p * (log(p) / log(2.0));
        }
    }
    return entropy;
}
