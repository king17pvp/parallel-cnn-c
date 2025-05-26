#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cnn.h"
#include "cnn_mpi.h"
#include <string.h>
#include "utils.h"
#include "mpi.h"

Tensor3D conv_forward_mpi_by_row(Tensor3D input, ConvLayer *layer, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int out_w = input.width - layer->kernel_size + 1;
    int out_h = input.height - layer->kernel_size + 1;
    int out_ch = layer->out_channels;
    int in_ch = layer->in_channels;
    int ks = layer->kernel_size;
    int total_out = out_w * out_h * out_ch;

    // Determine rows per process
    int rows_per_proc = out_h / size;
    int rem = out_h % size;
    int my_rows = (rank < rem) ? rows_per_proc + 1 : rows_per_proc;
    int start_row = rank * rows_per_proc + (rank < rem ? rank : rem);

    // Allocate output for this process
    float *my_output = malloc(sizeof(float) * my_rows * out_w * out_ch);

    for (int oc = 0; oc < out_ch; oc++) {
        for (int i = 0; i < my_rows; i++) {
            int global_i = start_row + i;
            for (int j = 0; j < out_w; j++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ki = 0; ki < ks; ki++) {
                        for (int kj = 0; kj < ks; kj++) {
                            int in_y = global_i + ki;
                            int in_x = j + kj;
                            int in_idx = ic * input.height * input.width + in_y * input.width + in_x;
                            int w_idx = oc * in_ch * ks * ks + ic * ks * ks + ki * ks + kj;
                            sum += input.data[in_idx] * layer->weights[w_idx];
                        }
                    }
                }
                int out_idx = oc * my_rows * out_w + i * out_w + j;
                my_output[out_idx] = relu(sum + layer->biases[oc]);
            }
        }
    }

    // Prepare to gather results to rank 0
    float *output_data = NULL;
    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        output_data = malloc(sizeof(float) * total_out);
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int r_rows = (r < rem) ? rows_per_proc + 1 : rows_per_proc;
            recvcounts[r] = r_rows * out_w * out_ch;
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    MPI_Gatherv(
        my_output,
        my_rows * out_w * out_ch,
        MPI_FLOAT,
        output_data,
        recvcounts,
        displs,
        MPI_FLOAT,
        0,
        comm
    );

    free(input.data);
    free(my_output);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
        return (Tensor3D){ out_w, out_h, out_ch, output_data };
    } else {
        return (Tensor3D){0, 0, 0, NULL};
    }
}
Tensor3D conv_forward_mpi(Tensor3D input, ConvLayer *layer, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int out_w = input.width - layer->kernel_size + 1;
    int out_h = input.height - layer->kernel_size + 1;
    int out_ch = layer->out_channels;
    int in_ch = layer->in_channels;
    int ks = layer->kernel_size;
    int total_out = out_w * out_h * out_ch;

    int channels_per_proc = out_ch / size;
    int rem = out_ch % size;
    int my_channels = (rank < rem) ? channels_per_proc + 1 : channels_per_proc;
    int start_channel = rank * channels_per_proc + (rank < rem ? rank : rem);

    float *my_output = malloc(sizeof(float) * my_channels * out_w * out_h);

    for (int oc = 0; oc < my_channels; oc++) {
        int global_oc = start_channel + oc;
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ki = 0; ki < ks; ki++) {
                        for (int kj = 0; kj < ks; kj++) {
                            int in_y = i + ki;
                            int in_x = j + kj;
                            int in_idx = ic * input.height * input.width + in_y * input.width + in_x;
                            int w_idx = global_oc * in_ch * ks * ks + ic * ks * ks + ki * ks + kj;
                            sum += input.data[in_idx] * layer->weights[w_idx];
                        }
                    }
                }
                int out_idx = oc * out_h * out_w + i * out_w + j;
                my_output[out_idx] = leaky_relu(sum + layer->biases[global_oc]);
            }
        }
    }

    float *output_data = NULL;
    if (rank == 0) output_data = malloc(sizeof(float) * total_out);

    int *recvcounts = NULL, *displs = NULL;
    if (rank == 0) {
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int ch = (r < rem) ? channels_per_proc + 1 : channels_per_proc;
            recvcounts[r] = ch * out_w * out_h;
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    MPI_Gatherv(
        my_output,
        my_channels * out_w * out_h,
        MPI_FLOAT,
        output_data,
        recvcounts,
        displs,
        MPI_FLOAT,
        0,
        comm
    );

    free(input.data);
    free(my_output);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
        return (Tensor3D){ out_w, out_h, out_ch, output_data };
    } else {
        return (Tensor3D){0, 0, 0, NULL};
    }
}

Tensor3D maxpool_forward_mpi_by_row(Tensor3D input, int pool_size, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int out_w = input.width / pool_size;
    int out_h = input.height / pool_size;
    int out_ch = input.channels;
    int total_out = out_w * out_h * out_ch;

    // Distribute rows of each channel across processes
    int rows_per_proc = out_h / size;
    int rem = out_h % size;
    int my_rows = (rank < rem) ? rows_per_proc + 1 : rows_per_proc;
    int start_row = rank * rows_per_proc + (rank < rem ? rank : rem);

    float *my_output = malloc(sizeof(float) * my_rows * out_w * out_ch);

    for (int c = 0; c < out_ch; c++) {
        for (int i = 0; i < my_rows; i++) {
            int global_i = start_row + i;
            for (int j = 0; j < out_w; j++) {
                float max = -1e9;
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int in_y = global_i * pool_size + pi;
                        int in_x = j * pool_size + pj;
                        int in_idx = c * input.height * input.width + in_y * input.width + in_x;
                        if (input.data[in_idx] > max)
                            max = input.data[in_idx];
                    }
                }
                int out_idx = c * my_rows * out_w + i * out_w + j;
                my_output[out_idx] = max;
            }
        }
    }

    // Allocate output on root
    float *output_data = NULL;
    int *recvcounts = NULL, *displs = NULL;

    if (rank == 0) {
        output_data = malloc(sizeof(float) * total_out);
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int r_rows = (r < rem) ? rows_per_proc + 1 : rows_per_proc;
            recvcounts[r] = r_rows * out_w * out_ch;
            displs[r] = offset;
            offset += recvcounts[r];
        }
    }

    MPI_Gatherv(
        my_output,
        my_rows * out_w * out_ch,
        MPI_FLOAT,
        output_data,
        recvcounts,
        displs,
        MPI_FLOAT,
        0,
        comm
    );

    free(input.data);
    free(my_output);
    if (rank == 0) {
        free(recvcounts);
        free(displs);
        return (Tensor3D){ out_w, out_h, out_ch, output_data };
    } else {
        return (Tensor3D){ 0, 0, 0, NULL };
    }
}

Vector fc_forward_mpi(Vector input, FullyConnectedLayer *layer, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int out_features = layer->out_features;
    int in_features = input.size;

    // Divide output neurons across processes
    int outputs_per_proc = out_features / size;
    int rem = out_features % size;
    int my_outputs = (rank < rem) ? outputs_per_proc + 1 : outputs_per_proc;
    int start_idx = rank * outputs_per_proc + (rank < rem ? rank : rem);

    float *my_output_data = malloc(sizeof(float) * my_outputs);

    for (int i = 0; i < my_outputs; i++) {
        int global_i = start_idx + i;
        float sum = 0.0f;
        for (int j = 0; j < in_features; j++) {
            sum += layer->weights[global_i * in_features + j] * input.data[j];
        }
        my_output_data[i] = leaky_relu(sum + layer->biases[global_i]);
    }

    float *output_data = NULL;
    int *recvcounts = NULL, *displs = NULL;

    if (rank == 0) {
        output_data = malloc(sizeof(float) * out_features);
        recvcounts = malloc(sizeof(int) * size);
        displs = malloc(sizeof(int) * size);
        int offset = 0;
        for (int r = 0; r < size; r++) {
            int count = (r < rem) ? outputs_per_proc + 1 : outputs_per_proc;
            recvcounts[r] = count;
            displs[r] = offset;
            offset += count;
        }
    }

    MPI_Gatherv(
        my_output_data,
        my_outputs,
        MPI_FLOAT,
        output_data,
        recvcounts,
        displs,
        MPI_FLOAT,
        0,
        comm
    );

    free(my_output_data);
    free(input.data);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
        return (Vector){ out_features, output_data };
    } else {
        return (Vector){ 0, NULL };
    }
}

void cnn_forward_mpi(CNN *cnn, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int total_input = cnn->input_width * cnn->input_height * cnn->input_channels;
    float *copy = malloc(sizeof(float) * total_input);
    for (int i = 0; i < total_input; i++) {
        copy[i] = cnn->input_data[i];
    }

    Tensor3D x = { cnn->input_width, cnn->input_height, cnn->input_channels, copy };

    // Convolution + Maxpool layers (MPI parallelized)
    for (int i = 0; i < cnn->num_conv_layers; i++) {
        x = conv_forward_mpi_by_row(x, &cnn->conv_layers[i], comm);
        // x = maxpool_forward_mpi_by_row(x, 2, comm);
    }

    Vector v;
    int flat_size = 0;
    if (rank == 0) {
        v = flatten(x);
        flat_size = v.size;
    }

    MPI_Bcast(&flat_size, 1, MPI_INT, 0, comm);

    if (rank != 0) {
        v.size = flat_size;
        v.data = malloc(sizeof(float) * flat_size);
    }

    // Broadcast vector data
    MPI_Bcast(v.data, flat_size, MPI_FLOAT, 0, comm);

    for (int i = 0; i < cnn->num_fc_layers; i++) {
        v = fc_forward_mpi(v, &cnn->fc_layers[i], comm);
    }

    if (rank == 0) {
        cnn->output = v.data[0];
    }

    free(v.data);
}

