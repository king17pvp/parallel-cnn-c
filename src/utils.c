#include "utils.h"
#include <stdlib.h>
#include <math.h>
float relu(float x) {
    return x > 0 ? x : 0;
}

// Box-Mullter transform
float rand_normal(float mean, float stddev) {
    float u1 = ((float) rand() + 1) / ((float) RAND_MAX + 2);  // avoid log(0)
    float u2 = ((float) rand() + 1) / ((float) RAND_MAX + 2);
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2 * M_PI * u2);
    return z0 * stddev + mean;
}