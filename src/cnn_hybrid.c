#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cnn.h"
#include "cnn_cuda.h"
#include "utils.h"
#include "mpi.h"

// Process a batch of images on each GPU
void process_image_batch(float *images, int num_images, CNN *cnn, float *results, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Each rank processes its batch of images
    for (int i = 0; i < num_images; i++) {
        // Create a CNN instance for this image
        CNN img_cnn = *cnn;  // Copy the CNN structure
        int input_size = cnn->input_width * cnn->input_height * cnn->input_channels;
        img_cnn.input_data = &images[i * input_size];
        
        // Process the image on GPU using existing CUDA implementation
        cnn_forward_cuda(&img_cnn);
        results[i] = img_cnn.output;
    }
}

// Main distributed inference function
void cnn_forward_hybrid(float *all_images, int total_images, CNN *cnn, float *all_results, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Calculate distribution of images across processes
    int images_per_proc = total_images / size;
    int remainder = total_images % size;
    int my_num_images = images_per_proc + (rank < remainder ? 1 : 0);
    int my_start_idx = rank * images_per_proc + (rank < remainder ? rank : remainder);

    // Calculate input size per image
    int input_size = cnn->input_width * cnn->input_height * cnn->input_channels;
    
    // Allocate memory for local images and results
    float *my_images = malloc(sizeof(float) * my_num_images * input_size);
    float *my_results = malloc(sizeof(float) * my_num_images);

    // Distribute images to processes
    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int num = images_per_proc + (r < remainder ? 1 : 0);
            recvcounts[r] = num * input_size;
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    // Scatter images to processes
    MPI_Scatterv(all_images, recvcounts, displs, MPI_FLOAT,
                 my_images, my_num_images * input_size, MPI_FLOAT,
                 0, comm);

    // Process local batch of images
    process_image_batch(my_images, my_num_images, cnn, my_results, comm);

    // Gather results from all processes
    if (rank == 0) {
        // Update receive counts and displacements for results
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int num = images_per_proc + (r < remainder ? 1 : 0);
            recvcounts[r] = num;
            displs[r] = offset;
            offset += num;
        }
    }

    // Gather results
    MPI_Gatherv(my_results, my_num_images, MPI_FLOAT,
                all_results, recvcounts, displs, MPI_FLOAT,
                0, comm);

    // Clean up
    free(my_images);
    free(my_results);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }
}
