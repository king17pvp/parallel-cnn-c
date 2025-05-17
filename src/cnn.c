// cnn.c
#include <stdlib.h>
#include <math.h>
#include "cnn.h"
#include "utils.h"

void add_conv_layer(CNN *cnn, int kernel_size, float stddev) {
    ConvLayer *layer = &cnn->conv_layers[cnn->num_conv_layers++];
    int size = kernel_size * kernel_size;
    layer->kernel_size = kernel_size;
    layer->weights = malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) layer->weights[i] = rand_normal(0.0f, stddev);
    layer->bias = rand_normal(0.0f, 0.1f);
}

void add_fc_layer(CNN *cnn, int in_features, int out_features, float stddev) {
    FullyConnectedLayer *layer = &cnn->fc_layers[cnn->num_fc_layers++];
    layer->in_features = in_features;
    layer->out_features = out_features;
    layer->weights = malloc(sizeof(float) * in_features * out_features);
    layer->biases = malloc(sizeof(float) * out_features);
    for (int i = 0; i < in_features * out_features; i++)
        layer->weights[i] = rand_normal(0.0f, stddev);
    for (int i = 0; i < out_features; i++)
        layer->biases[i] = rand_normal(0.0f, stddev);
}

Image2D conv_forward(Image2D input, ConvLayer *layer) {
    int out_w = input.width - layer->kernel_size + 1;
    int out_h = input.height - layer->kernel_size + 1;
    float *out_data = malloc(sizeof(float) * out_w * out_h);

    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < layer->kernel_size; ki++) {
                for (int kj = 0; kj < layer->kernel_size; kj++) {
                    int in_idx = (i + ki) * input.width + (j + kj);
                    int w_idx = ki * layer->kernel_size + kj;
                    sum += input.data[in_idx] * layer->weights[w_idx];
                }
            }
            out_data[i * out_w + j] = relu(sum + layer->bias);
        }
    }

    free(input.data);
    return (Image2D){ out_w, out_h, out_data };
}

Image2D maxpool_forward(Image2D input, int pool_size) {
    int out_w = input.width / pool_size;
    int out_h = input.height / pool_size;
    float *out_data = malloc(sizeof(float) * out_w * out_h);

    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            float max = -1e9;
            for (int pi = 0; pi < pool_size; pi++) {
                for (int pj = 0; pj < pool_size; pj++) {
                    int x = j * pool_size + pj;
                    int y = i * pool_size + pi;
                    float val = input.data[y * input.width + x];
                    if (val > max) max = val;
                }
            }
            out_data[i * out_w + j] = max;
        }
    }

    free(input.data);
    return (Image2D){ out_w, out_h, out_data };
}

Vector flatten(Image2D input) {
    int total = input.width * input.height;
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
        out.data[i] = relu(sum + layer->biases[i]);
    }
    free(input.data);
    return out;
}

void cnn_forward(CNN *cnn) {
    Image2D x = { cnn->input_size, cnn->input_size, cnn->input_data };

    for (int i = 0; i < cnn->num_conv_layers; i++) {
        x = conv_forward(x, &cnn->conv_layers[i]);
        x = maxpool_forward(x, 2);
    }

    Vector v = flatten(x);

    for (int i = 0; i < cnn->num_fc_layers; i++) {
        v = fc_forward(v, &cnn->fc_layers[i]);
    }

    cnn->output = v.data[0];
    free(v.data);
}
