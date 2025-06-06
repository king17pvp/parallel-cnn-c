#ifndef UTILS_H
#define UTILS_H
#include "cnn.h"
#include <stdio.h>
float relu(float x);
float leaky_relu(float x);
float rand_normal(float mean, float stddev);
void add_conv_layer(CNN *cnn, int out_channels, int kernel_size, int in_channels, float mean, float stddev);
void add_fc_layer(CNN *cnn, int in_features, int out_features, float mean, float stddev);
void save_cnn_weights(CNN *cnn, const char *filename);
void load_cnn_weights(CNN *cnn, const char *filename);
void load_config_from_txt(const char *filename, int *input_w, int *input_h, int *input_c,
                          int *num_conv, int *ks, int *hidden, float *mean, float *std, int *max_pool_stride);
#endif
