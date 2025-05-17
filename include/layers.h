#ifndef LAYERS_H
#define LAYERS_H

#include <stdlib.h>

// General 2D image/tensor
typedef struct {
    int width;
    int height;
    int depth;      // for future extension (channels)
    float *data;    // size = width * height
} Image2D;

// Convolutional Layer
typedef struct {
    int kernel_size;
    int input_size;
    int output_size;
    float *weights; // [kernel_size * kernel_size]
    float bias;
} ConvLayer;

// Fully Connected Layer
typedef struct {
    int in_features;
    int out_features;
    float *weights; // [out_features * in_features]
    float *biases;  // [out_features]
} FullyConnectedLayer;

// 1D vector
typedef struct {
    int size;
    float *data;
} Vector;

// Function declarations
float relu(float x);
float rand_normal(float mean, float stddev);

Image2D conv_forward(Image2D input, ConvLayer *layer);
Image2D maxpool_forward(Image2D input, int pool_size);
Vector flatten(Image2D input);
Vector fc_forward(Vector input, FullyConnectedLayer *layer);

#endif
#ifndef CNN_H
#define CNN_H

#include "layers.h"

#define MAX_CONV_LAYERS 10
#define MAX_FC_LAYERS 10

typedef struct {
    ConvLayer conv_layers[MAX_CONV_LAYERS];
    int num_conv_layers;

    FullyConnectedLayer fc_layers[MAX_FC_LAYERS];
    int num_fc_layers;

    int input_size;
    float *input_data;

    float output;
} CNN;

// Forward declarations
void cnn_forward(CNN *cnn);
void add_conv_layer(CNN *cnn, int kernel_size, float stddev);
void add_fc_layer(CNN *cnn, int in_features, int out_features, float stddev);

#endif
