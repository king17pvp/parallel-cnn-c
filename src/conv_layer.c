#include "layers.h"
#include "utils.h"

void conv_forward(Input *input, ConvLayer *layer, ConvOutput *output) {
    int conv_size = INPUT_SIZE - FILTER_SIZE + 1;

    for (int i = 0; i < conv_size; i++) {
        for (int j = 0; j < conv_size; j++) {
            float sum = 0.0;
            for (int fi = 0; fi < FILTER_SIZE; fi++) {
                for (int fj = 0; fj < FILTER_SIZE; fj++) {
                    int in_idx = (i + fi) * INPUT_SIZE + (j + fj);
                    int f_idx = fi * FILTER_SIZE + fj;
                    sum += input->data[in_idx] * layer->weights[f_idx];
                }
            }
            output->data[i * conv_size + j] = relu(sum + layer->bias);
        }
    }
}
