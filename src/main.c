#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"
#include "utils.h"

int main() {
    srand(time(NULL));
    CNN cnn;

    cnn.input.data = malloc(sizeof(float) * INPUT_SIZE * INPUT_SIZE);
    cnn.conv.weights = malloc(sizeof(float) * FILTER_SIZE * FILTER_SIZE);
    cnn.conv_out.data = malloc(sizeof(float) * (INPUT_SIZE - FILTER_SIZE + 1) * (INPUT_SIZE - FILTER_SIZE + 1));
    cnn.pool_out.data = malloc(sizeof(float) * FC_INPUT_SIZE);
    cnn.flat_out.data = malloc(sizeof(float) * FC_INPUT_SIZE);
    cnn.fc.weights = malloc(sizeof(float) * FC_OUTPUT_SIZE * FC_INPUT_SIZE);
    cnn.fc.biases = malloc(sizeof(float) * FC_OUTPUT_SIZE);

    // Fill with test data (manually or via file)
    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE; i++) cnn.input.data[i] = (i < 27) ? 0.5f : 0.2f;
    for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++) cnn.conv.weights[i] = rand_normal(0.0f, 1.0f);
    cnn.conv.bias = rand_normal(0.0f, 1.0f);
    for (int i = 0; i < FC_OUTPUT_SIZE * FC_INPUT_SIZE; i++) cnn.fc.weights[i] = rand_normal(0.0f, 1.0f);
    for (int i = 0; i < FC_OUTPUT_SIZE; i++) cnn.fc.biases[i] = rand_normal(0.0f, 1.0f);

    cnn_forward(&cnn);
    printf("CNN Output: %f\n", cnn.output[0]);
    printf("Conv[0]: %f\n", cnn.conv_out.data[0]);
    printf("Pool[0]: %f\n", cnn.pool_out.data[0]);
    printf("Flat[0]: %f\n", cnn.flat_out.data[0]);

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
