#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cnn.h"
#include "cnn_cuda.h"
#include "utils.h"
#include <string.h>

int main(int argc, char **argv) {
    srand(time(NULL));
    
    CNN cnn = {0};
    cnn.input_width = 2048;
    cnn.input_height = 2048;
    cnn.input_channels = 3;

    int kernel_size = 5;
    int max_pool_stride = 1;
    int hidden_dim = 128;
    int current_width = cnn.input_width;
    int current_height = cnn.input_height;
    int current_channels = cnn.input_channels;
    float mean = 0.0f;
    float std = 1.0f;
    int NUM_CONV_LAYERS = 500;

    int input_volume = current_width * current_height * current_channels;
    cnn.input_data = malloc(sizeof(float) * input_volume);
    for (int i = 0; i < input_volume; i++)
        cnn.input_data[i] = rand_normal(mean, std);
    
    for (int i = 0; i < NUM_CONV_LAYERS; ++i) {
        int out_channels = 3; // increase depth
        add_conv_layer(&cnn, out_channels, kernel_size, current_channels, mean, std);
        current_width = (current_width - kernel_size + 1) / max_pool_stride;
        current_height = (current_height - kernel_size + 1) / max_pool_stride;
        current_channels = out_channels;
        printf("After conv layer %d: %d x %d x %d\n", i + 1, current_width, current_height, current_channels);
    }

    // Flatten size
    int flatten_size = current_width * current_height * current_channels;

    // Add fully connected layers
    int i = 0;
    int in_dim = flatten_size;
    while (hidden_dim >= 1) {
        add_fc_layer(&cnn, in_dim, hidden_dim, mean, std);
        printf("Vector after FC layer %d: %d -> %d\n", i + 1, in_dim, hidden_dim);
        in_dim = hidden_dim;
        hidden_dim /= 2;
        i++;
    }
    if (argc > 1 && strcmp(argv[1], "load") == 0) {
        load_cnn_weights(&cnn, "./ckpts/cnn_weights.bin");
    } else if (argc > 1 && strcmp(argv[1], "save") == 0) {
        save_cnn_weights(&cnn, "./ckpts/cnn_weights.bin");
    }
    clock_t start = clock();
    cnn_forward_cuda(&cnn);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Final output: %f\n", cnn.output);
    printf("Elapsed time using CUDA: %.6f seconds\n", elapsed);
    printf("Freeing memory\n");
    
    for (int i = 0; i < cnn.num_conv_layers; i++) {
        free(cnn.conv_layers[i].weights);
        free(cnn.conv_layers[i].biases);
    }
    for (int i = 0; i < cnn.num_fc_layers; i++) {
        free(cnn.fc_layers[i].weights);
        free(cnn.fc_layers[i].biases);
    }
    free(cnn.input_data);
    return 0;
}