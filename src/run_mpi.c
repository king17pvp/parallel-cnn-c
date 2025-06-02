#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cnn.h"
#include "cnn_mpi.h"
#include "utils.h"
#include "mpi.h"
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    srand(time(NULL));
    int input_w, input_h, input_c, num_conv, kernel_size, hidden_dim, max_pool_stride;
    float mean, std;
    load_config_from_txt("configs/config.txt", &input_w, &input_h, &input_c, &num_conv, &kernel_size, &hidden_dim, &mean, &std, &max_pool_stride);
    CNN cnn = {0};
    cnn.input_width = input_w;
    cnn.input_height = input_h;
    cnn.input_channels = input_c;
    int current_width = cnn.input_width;
    int current_height = cnn.input_height;
    int current_channels = cnn.input_channels;
    int NUM_CONV_LAYERS = num_conv;
    int input_volume = current_width * current_height * current_channels;
    printf("Normal %.2f, Std %.2f\n", mean, std);
    cnn.input_data = malloc(sizeof(float) * input_volume);
    for (int i = 0; i < input_volume; i++)
        cnn.input_data[i] = rand_normal(mean, std);
    
    for (int i = 0; i < NUM_CONV_LAYERS; ++i) {
        int out_channels = (i + 1) * 4; 
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (argc > 1 && strcmp(argv[1], "load") == 0) {
        if (rank == 0) load_cnn_weights(&cnn, "./ckpts/cnn_weights.bin");
        MPI_Barrier(MPI_COMM_WORLD); 
    } else if (argc > 1 && strcmp(argv[1], "save") == 0) {
        if (rank == 0) save_cnn_weights(&cnn, "./ckpts/cnn_weights.bin");
        MPI_Barrier(MPI_COMM_WORLD); 
    }
    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks start at the same time
    double start_mpi = MPI_Wtime();
    cnn_forward_mpi(&cnn, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks finish
    double end_mpi = MPI_Wtime();
    double elapsed = (double)(end_mpi - start_mpi);
    
    if (rank == 0) {
        printf("CNN output using MPI: %f\n", cnn.output);
        printf("Elapsed time using MPI: %.6f seconds\n", elapsed);
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