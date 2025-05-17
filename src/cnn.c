#include <stdlib.h>
#include <math.h>
#include "cnn.h"
#include "utils.h"

void add_conv_layer(CNN *cnn, int out_channels, int kernel_size, int in_channels, float stddev) {
    ConvLayer *layer = &cnn->conv_layers[cnn->num_conv_layers++];
    int kernel_area = kernel_size * kernel_size;
    int total_weights = out_channels * in_channels * kernel_area;
    layer->out_channels = out_channels;
    layer->in_channels = in_channels;
    layer->kernel_size = kernel_size;
    layer->weights = malloc(sizeof(float) * total_weights);
    layer->biases = malloc(sizeof(float) * out_channels);
    for (int i = 0; i < total_weights; i++)
        layer->weights[i] = rand_normal(0.0f, stddev);
    for (int i = 0; i < out_channels; i++)
        layer->biases[i] = rand_normal(0.0f, stddev);
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
                output_data[out_idx] = leaky_relu(sum + layer->biases[oc]);
            }
        }
    }

    free(input.data);
    return (Tensor3D){ out_w, out_h, out_ch, output_data };
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

void cnn_forward(CNN *cnn) {
    int total_input = cnn->input_width * cnn->input_height * cnn->input_channels;
    float *copy = malloc(sizeof(float) * total_input);
    for (int i = 0; i < total_input; i++) copy[i] = cnn->input_data[i];

    Tensor3D x = { cnn->input_width, cnn->input_height, cnn->input_channels, copy };

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
