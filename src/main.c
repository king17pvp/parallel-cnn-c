#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cnn.h"
#include "utils.h"
#include "mpi.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    srand(time(NULL));
    
    CNN cnn = {0};
    cnn.input_width = 512;
    cnn.input_height = 512;
    cnn.input_channels = 3;

    int kernel_size = 5;
    int max_pool_stride = 2;
    int hidden_dim = 128;
    int current_width = cnn.input_width;
    int current_height = cnn.input_height;
    int current_channels = cnn.input_channels;
    float mean = 0.0f;
    float std = 1.0f;
    int NUM_CONV_LAYERS = 4;

    int input_volume = current_width * current_height * current_channels;
    cnn.input_data = malloc(sizeof(float) * input_volume);
    for (int i = 0; i < input_volume; i++)
        cnn.input_data[i] = rand_normal(mean, std);

    // Add convolutional layers
    
    for (int i = 0; i < NUM_CONV_LAYERS; ++i) {
        int out_channels = (i + 1) * 4; // increase depth
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
    clock_t start = clock();
    cnn_forward(&cnn);
    printf("CNN Output: %f\n", cnn.output);
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f seconds\n", elapsed);

    start = clock();
    cnn_forward(&cnn);
    printf("CNN Output: %f\n", cnn.output);
    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f seconds\n", elapsed);
    start = clock();
    cnn_forward_mpi(&cnn, MPI_COMM_WORLD);
    end = clock();
    elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f seconds\n", elapsed);
    int rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("Final CNN output using MPI: %f\n", cnn.output);
    }

    MPI_Finalize();
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