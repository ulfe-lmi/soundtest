#include "dct.h"
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void dct_ii(const double *in, double *out, size_t n) {
    double factor = sqrt(2.0 / n);
    for (size_t k = 0; k < n; ++k) {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            sum += in[i] * cos(M_PI / n * (i + 0.5) * k);
        }
        out[k] = factor * sum;
    }
    out[0] /= sqrt(2.0);
}

void idct_ii(const double *in, double *out, size_t n) {
    double factor = sqrt(2.0 / n);
    for (size_t i = 0; i < n; ++i) {
        double sum = in[0] / sqrt(2.0);
        for (size_t k = 1; k < n; ++k) {
            sum += in[k] * cos(M_PI / n * (i + 0.5) * k);
        }
        out[i] = factor * sum;
    }
}
