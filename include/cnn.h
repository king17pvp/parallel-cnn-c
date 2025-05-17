#ifndef CNN_H
#define CNN_H

#define MAX_CONV_LAYERS 10
#define MAX_FC_LAYERS 10

typedef struct {
    int kernel_size;
    float *weights;
    float bias;
} ConvLayer;

typedef struct {
    int in_features;
    int out_features;
    float *weights;
    float *biases;
} FullyConnectedLayer;

typedef struct {
    int width, height;
    float *data;
} Image2D;

typedef struct {
    int size;
    float *data;
} Vector;

typedef struct {
    ConvLayer conv_layers[MAX_CONV_LAYERS];
    int num_conv_layers;
    FullyConnectedLayer fc_layers[MAX_FC_LAYERS];
    int num_fc_layers;
    int input_size;
    float *input_data;
    float output;
} CNN;

void add_conv_layer(CNN *cnn, int kernel_size, float stddev);
void add_fc_layer(CNN *cnn, int in_features, int out_features, float stddev);
void cnn_forward(CNN *cnn);

#endif
