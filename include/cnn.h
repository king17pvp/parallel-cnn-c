#ifndef CNN_H
#define CNN_H

#define MAX_CONV_LAYERS 10
#define MAX_FC_LAYERS 10

// 3D image/tensor
typedef struct {
    int width, height, channels;
    float *data; // size = width * height * channels
} Tensor3D;

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    float *weights; // size = out_channels * in_channels * kernel_size * kernel_size
    float *biases;  // size = out_channels
} ConvLayer;

typedef struct {
    int in_features;
    int out_features;
    float *weights; // [out_features * in_features]
    float *biases;  // [out_features]
} FullyConnectedLayer;

typedef struct {
    int size;
    float *data;
} Vector;

typedef struct {
    ConvLayer conv_layers[MAX_CONV_LAYERS];
    int num_conv_layers;
    FullyConnectedLayer fc_layers[MAX_FC_LAYERS];
    int num_fc_layers;

    int input_width;
    int input_height;
    int input_channels;
    float *input_data;

    float output;
} CNN;

void add_conv_layer(CNN *cnn, int out_channels, int kernel_size, int in_channels, float stddev);
void add_fc_layer(CNN *cnn, int in_features, int out_features, float stddev);
void cnn_forward(CNN *cnn);

#endif
