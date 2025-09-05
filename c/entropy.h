#ifndef ENTROPY_H
#define ENTROPY_H

#include <stddef.h>
#include <stdint.h>

double calculate_byte_entropy(const int8_t *data, size_t length);
double calculate_16bit_entropy(const int16_t *data, size_t length);

#endif // ENTROPY_H
