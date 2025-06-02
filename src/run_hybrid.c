#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cnn.h"
#include "cnn_hybrid.h"
#include "utils.h"
#include "mpi.h"
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(NULL));
    int input_w, input_h, input_c, num_conv, kernel_size, hidden_dim, max_pool_stride;
    float mean, std;
    
    // Load CNN configuration
    load_config_from_txt("configs/config.txt", &input_w, &input_h, &input_c, 
                        &num_conv, &kernel_size, &hidden_dim, &mean, &std, &max_pool_stride);
    
    // Initialize CNN
    CNN cnn = {0};
    cnn.input_width = input_w;
    cnn.input_height = input_h;
    cnn.input_channels = input_c;
    int current_width = cnn.input_width;
    int current_height = cnn.input_height;
    int current_channels = cnn.input_channels;
    int NUM_CONV_LAYERS = num_conv;
    
    // Add convolutional layers
    for (int i = 0; i < NUM_CONV_LAYERS; ++i) {
        int out_channels = (i + 1) * 4;
        add_conv_layer(&cnn, out_channels, kernel_size, current_channels, mean, std);
        current_width = (current_width - kernel_size + 1) / max_pool_stride;
        current_height = (current_height - kernel_size + 1) / max_pool_stride;
        current_channels = out_channels;
        if (rank == 0) {
            printf("After conv layer %d: %d x %d x %d\n", i + 1, current_width, current_height, current_channels);
        }
    }

    // Get flatten size for FC layers
    int flatten_size = current_width * current_height * current_channels;
    
    // Add fully connected layers
    int i = 0;
    int in_dim = flatten_size;
    while (hidden_dim >= 1) {
        add_fc_layer(&cnn, in_dim, hidden_dim, mean, std);
        if (rank == 0) {
            printf("Vector after FC layer %d: %d -> %d\n", i + 1, in_dim, hidden_dim);
        }
        in_dim = hidden_dim;
        hidden_dim /= 2;
        i++;
    }

    // Load weights if requested
    if (argc > 1 && strcmp(argv[1], "load") == 0) {
        if (rank == 0) {
            load_cnn_weights(&cnn, "./ckpts/cnn_weights.bin");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Prepare multiple test images
    int num_images = 16;  // Process 16 images in parallel
    int input_size = input_w * input_h * input_c;
    float *all_images = NULL;
    float *all_results = NULL;
    
    if (rank == 0) {
        all_images = malloc(sizeof(float) * num_images * input_size);
        all_results = malloc(sizeof(float) * num_images);
        
        // Generate random test images
        for (int img = 0; img < num_images; img++) {
            for (int i = 0; i < input_size; i++) {
                all_images[img * input_size + i] = rand_normal(mean, std);
            }
        }
    }

    // Process images using hybrid MPI+CUDA implementation
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    cnn_forward_hybrid(all_images, num_images, &cnn, all_results, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Processed %d images\n", num_images);
        for (int i = 0; i < num_images; i++) {
            printf("Image %d output: %f\n", i, all_results[i]);
        }
        printf("Total time for %d images: %.6f seconds\n", num_images, end_time - start_time);
        printf("Average time per image: %.6f seconds\n", (end_time - start_time) / num_images);
        
        // Clean up
        free(all_images);
        free(all_results);
    }

    // Save weights if requested
    if (argc > 1 && strcmp(argv[1], "save") == 0) {
        if (rank == 0) {
            save_cnn_weights(&cnn, "./ckpts/cnn_weights.bin");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Clean up CNN
    for (int i = 0; i < cnn.num_conv_layers; i++) {
        free(cnn.conv_layers[i].weights);
        free(cnn.conv_layers[i].biases);
    }
    for (int i = 0; i < cnn.num_fc_layers; i++) {
        free(cnn.fc_layers[i].weights);
        free(cnn.fc_layers[i].biases);
    }

    MPI_Finalize();
    return 0;
}
