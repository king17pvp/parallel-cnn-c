#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"

int main() {
    CNN cnn;

    cnn.input.data = malloc(sizeof(float) * INPUT_SIZE * INPUT_SIZE);
    cnn.conv.weights = malloc(sizeof(float) * FILTER_SIZE * FILTER_SIZE);
    cnn.conv_out.data = malloc(sizeof(float) * (INPUT_SIZE - FILTER_SIZE + 1) * (INPUT_SIZE - FILTER_SIZE + 1));
    cnn.pool_out.data = malloc(sizeof(float) * FC_INPUT_SIZE);
    cnn.flat_out.data = malloc(sizeof(float) * FC_INPUT_SIZE);
    cnn.fc.weights = malloc(sizeof(float) * FC_OUTPUT_SIZE * FC_INPUT_SIZE);
    cnn.fc.biases = malloc(sizeof(float) * FC_OUTPUT_SIZE);

    // Fill with test data (manually or via file)
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) cnn.input.data[i] = (i < 27) ? 1 : 0;
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) cnn.conv.weights[i] = (i % 3 == 0) ? 1 : -1;
    cnn.conv.bias = 0;
    for (int i = 0; i < FC_OUTPUT_SIZE * FC_INPUT_SIZE; i++) cnn.fc.weights[i] = 0.1f * (i + 1);
    cnn.fc.biases[0] = 0.5;

    cnn_forward(&cnn);
    printf("CNN Output: %f\n", cnn.output[0]);

    // Free memory
    free(cnn.input.data);
    free(cnn.conv.weights);
    free(cnn.conv_out.data);
    free(cnn.pool_out.data);
    free(cnn.flat_out.data);
    free(cnn.fc.weights);
    free(cnn.fc.biases);

    return 0;
}
