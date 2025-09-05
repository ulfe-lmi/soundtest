#ifndef DCT_H
#define DCT_H

#include <stddef.h>

void dct_ii(const double *in, double *out, size_t n);
void idct_ii(const double *in, double *out, size_t n);

#endif // DCT_H
