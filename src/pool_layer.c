#include "layers.h"

void maxpool_forward(ConvOutput *input, PoolOutput *output) {
    int conv_size = INPUT_SIZE - FILTER_SIZE + 1;
    int pool_size = conv_size / POOL_SIZE;

    for (int i = 0; i < conv_size; i += POOL_SIZE) {
        for (int j = 0; j < conv_size; j += POOL_SIZE) {
            float max = input->data[i * conv_size + j];
            for (int pi = 0; pi < POOL_SIZE; pi++) {
                for (int pj = 0; pj < POOL_SIZE; pj++) {
                    int idx = (i + pi) * conv_size + (j + pj);
                    if (input->data[idx] > max)
                        max = input->data[idx];
                }
            }
            int out_i = i / POOL_SIZE;
            int out_j = j / POOL_SIZE;
            output->data[out_i * pool_size + out_j] = max;
        }
    }
}
