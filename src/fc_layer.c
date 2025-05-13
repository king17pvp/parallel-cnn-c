#include "layers.h"
#include "utils.h"

void fc_forward(FlattenOutput *input, FullyConnectedLayer *fc, float *output) {
    for (int i = 0; i < FC_OUTPUT_SIZE; i++) {
        float sum = 0.0;
        for (int j = 0; j < FC_INPUT_SIZE; j++) {
            sum += fc->weights[i * FC_INPUT_SIZE + j] * input->data[j];
        }
        output[i] = relu(sum + fc->biases[i]);
    }
}
