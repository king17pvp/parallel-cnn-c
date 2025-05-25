#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cnn.h"
#include <string.h>
#include "utils.h"
#include "mpi.h"
void add_conv_layer(CNN *cnn, int out_channels, int kernel_size, int in_channels, float mean, float stddev) {
    ConvLayer *layer = &cnn->conv_layers[cnn->num_conv_layers++];
    int kernel_area = kernel_size * kernel_size;
    int total_weights = out_channels * in_channels * kernel_area;
    layer->out_channels = out_channels;
    layer->in_channels = in_channels;
    layer->kernel_size = kernel_size;
    layer->weights = malloc(sizeof(float) * total_weights);
    layer->biases = malloc(sizeof(float) * out_channels);
    for (int i = 0; i < total_weights; i++)
        layer->weights[i] = rand_normal(mean, stddev);
    for (int i = 0; i < out_channels; i++)
        layer->biases[i] = rand_normal(mean, stddev);
}

void add_fc_layer(CNN *cnn, int in_features, int out_features, float mean, float stddev) {
    FullyConnectedLayer *layer = &cnn->fc_layers[cnn->num_fc_layers++];
    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->weights = malloc(sizeof(float) * in_features * out_features);
    layer->biases = malloc(sizeof(float) * out_features);
    for (int i = 0; i < in_features * out_features; i++)
        layer->weights[i] = rand_normal(mean, stddev);
    for (int i = 0; i < out_features; i++)
        layer->biases[i] = rand_normal(mean, stddev);
}

Tensor3D conv_forward(Tensor3D input, ConvLayer *layer) {
    int out_w = input.width - layer->kernel_size + 1;
    int out_h = input.height - layer->kernel_size + 1;
    int out_ch = layer->out_channels;
    int in_ch = layer->in_channels;
    int ks = layer->kernel_size;
    float *output_data = malloc(sizeof(float) * out_w * out_h * out_ch);

    for (int oc = 0; oc < out_ch; oc++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int ki = 0; ki < ks; ki++) {
                        for (int kj = 0; kj < ks; kj++) {
                            int in_y = i + ki;
                            int in_x = j + kj;
                            int in_idx = ic * input.height * input.width + in_y * input.width + in_x;
                            int w_idx = oc * in_ch * ks * ks + ic * ks * ks + ki * ks + kj;
                            sum += input.data[in_idx] * layer->weights[w_idx];
                        }
                    }
                }
                int out_idx = oc * out_h * out_w + i * out_w + j;
                output_data[out_idx] = relu(sum + layer->biases[oc]);
            }
        }
    }

    free(input.data);
    return (Tensor3D){ out_w, out_h, out_ch, output_data };
}
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

    // Use Gatherv to collect data from all processes
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
Tensor3D maxpool_forward(Tensor3D input, int pool_size) {
    int out_w = input.width / pool_size;
    int out_h = input.height / pool_size;
    int out_ch = input.channels;
    float *output_data = malloc(sizeof(float) * out_w * out_h * out_ch);

    for (int c = 0; c < out_ch; c++) {
        for (int i = 0; i < out_h; i++) {
            for (int j = 0; j < out_w; j++) {
                float max = -1e9;
                for (int pi = 0; pi < pool_size; pi++) {
                    for (int pj = 0; pj < pool_size; pj++) {
                        int in_y = i * pool_size + pi;
                        int in_x = j * pool_size + pj;
                        int in_idx = c * input.height * input.width + in_y * input.width + in_x;
                        if (input.data[in_idx] > max)
                            max = input.data[in_idx];
                    }
                }
                int out_idx = c * out_h * out_w + i * out_w + j;
                output_data[out_idx] = max;
            }
        }
    }

    free(input.data);
    return (Tensor3D){ out_w, out_h, out_ch, output_data };
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

Vector flatten(Tensor3D input) {
    int total = input.width * input.height * input.channels;
    Vector v = { total, malloc(sizeof(float) * total) };
    for (int i = 0; i < total; i++) v.data[i] = input.data[i];
    free(input.data);
    return v;
}

Vector fc_forward(Vector input, FullyConnectedLayer *layer) {
    Vector out = { layer->out_features, malloc(sizeof(float) * layer->out_features) };
    for (int i = 0; i < layer->out_features; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input.size; j++) {
            sum += layer->weights[i * input.size + j] * input.data[j];
        }
        out.data[i] = leaky_relu(sum + layer->biases[i]);
    }
    free(input.data);
    return out;
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


void cnn_forward(CNN *cnn) {
    int total_input = cnn->input_width * cnn->input_height * cnn->input_channels;
    float *copy = malloc(sizeof(float) * total_input);
    for (int i = 0; i < total_input; i++) copy[i] = cnn->input_data[i];

    Tensor3D x = { cnn->input_width, cnn->input_height, cnn->input_channels, copy };

    for (int i = 0; i < cnn->num_conv_layers; i++) {
        x = conv_forward(x, &cnn->conv_layers[i]);
        // x = maxpool_forward(x, 2);
    }

    Vector v = flatten(x);
    for (int i = 0; i < cnn->num_fc_layers; i++) {
        v = fc_forward(v, &cnn->fc_layers[i]);
    }

    cnn->output = v.data[0];
    free(v.data);
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

#define CHECK_FREAD(ptr, size, count, file) \
    if (fread(ptr, size, count, file) != count) { \
        fprintf(stderr, "Failed to read from file at %s:%d\n", __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }
#define CHECK_FWRITE(ptr, size, count, stream)                                      \
    do {                                                                            \
        size_t written = fwrite(ptr, size, count, stream);                          \
        if (written != (size_t)(count)) {                                           \
            fprintf(stderr, "fwrite error: expected %zu elements, wrote %zu\n",     \
                    (size_t)(count), written);                                      \
            perror("fwrite");                                                       \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)
void save_cnn_weights(CNN *cnn, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { perror("fopen"); exit(1); }

    fwrite(&cnn->num_conv_layers, sizeof(int), 1, f);
    for (int i = 0; i < cnn->num_conv_layers; ++i) {
        ConvLayer *l = &cnn->conv_layers[i];
        CHECK_FWRITE(&l->in_channels, sizeof(int), 1, f);
        CHECK_FWRITE(&l->out_channels, sizeof(int), 1, f);
        CHECK_FWRITE(&l->kernel_size, sizeof(int), 1, f);
        int w_len = l->in_channels * l->out_channels * l->kernel_size * l->kernel_size;
        CHECK_FWRITE(l->weights, sizeof(float), w_len, f);
        CHECK_FWRITE(l->biases, sizeof(float), l->out_channels, f);
    }

    CHECK_FWRITE(&cnn->num_fc_layers, sizeof(int), 1, f);
    for (int i = 0; i < cnn->num_fc_layers; ++i) {
        FullyConnectedLayer *fc = &cnn->fc_layers[i];
        CHECK_FWRITE(&fc->in_features, sizeof(int), 1, f);
        CHECK_FWRITE(&fc->out_features, sizeof(int), 1, f);
        int w_len = fc->in_features * fc->out_features;
        CHECK_FWRITE(fc->weights, sizeof(float), w_len, f);
        CHECK_FWRITE(fc->biases, sizeof(float), fc->out_features, f);
    }

    fclose(f);
}

void load_cnn_weights(CNN *cnn, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { perror("fopen"); exit(1); }

    CHECK_FREAD(&cnn->num_conv_layers, sizeof(int), 1, f);
    for (int i = 0; i < cnn->num_conv_layers; ++i) {
        ConvLayer *l = &cnn->conv_layers[i];
        CHECK_FREAD(&l->in_channels, sizeof(int), 1, f);
        CHECK_FREAD(&l->out_channels, sizeof(int), 1, f);
        CHECK_FREAD(&l->kernel_size, sizeof(int), 1, f);
        int w_len = l->in_channels * l->out_channels * l->kernel_size * l->kernel_size;
        l->weights = malloc(sizeof(float) * w_len);
        l->biases = malloc(sizeof(float) * l->out_channels);
        CHECK_FREAD(l->weights, sizeof(float), w_len, f);
        CHECK_FREAD(l->biases, sizeof(float), l->out_channels, f);
    }

    CHECK_FREAD(&cnn->num_fc_layers, sizeof(int), 1, f);
    for (int i = 0; i < cnn->num_fc_layers; ++i) {
        FullyConnectedLayer *fc = &cnn->fc_layers[i];
        CHECK_FREAD(&fc->in_features, sizeof(int), 1, f);
        CHECK_FREAD(&fc->out_features, sizeof(int), 1, f);
        int w_len = fc->in_features * fc->out_features;
        fc->weights = malloc(sizeof(float) * w_len);
        fc->biases = malloc(sizeof(float) * fc->out_features);
        CHECK_FREAD(fc->weights, sizeof(float), w_len, f);
        CHECK_FREAD(fc->biases, sizeof(float), fc->out_features, f);
    }

    fclose(f);
}