// main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cnn.h"
#include "utils.h"

int main() {
    srand(time(NULL));

    CNN cnn = {0};
    cnn.input_size = 8;
    cnn.input_data = malloc(sizeof(float) * cnn.input_size * cnn.input_size);
    for (int i = 0; i < cnn.input_size * cnn.input_size; i++)
        cnn.input_data[i] = rand_normal(0.0f, 1.0f);

    add_conv_layer(&cnn, 3, 0.2f);
    add_conv_layer(&cnn, 3, 0.2f);

    add_fc_layer(&cnn, 1, 8, 0.1f);
    add_fc_layer(&cnn, 8, 1, 0.1f);

    cnn_forward(&cnn);

    printf("CNN Output: %f\n", cnn.output);

    for (int i = 0; i < cnn.num_conv_layers; i++)
        free(cnn.conv_layers[i].weights);
    for (int i = 0; i < cnn.num_fc_layers; i++) {
        free(cnn.fc_layers[i].weights);
        free(cnn.fc_layers[i].biases);
    }
    free(cnn.input_data);

    return 0;
}
