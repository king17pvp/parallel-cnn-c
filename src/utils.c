#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "cnn.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>
float relu(float x) {
    return x > 0 ? x : 0;
}
float leaky_relu(float x) {
    return x > 0 ? x : 0.01f * x;
}
float rand_normal(float mean, float stddev) {
    srand(time(NULL));
    float u1 = ((float) rand() + 1) / ((float) RAND_MAX + 2);
    float u2 = ((float) rand() + 1) / ((float) RAND_MAX + 2);
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2 * M_PI * u2);
    return z0 * stddev + mean;
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