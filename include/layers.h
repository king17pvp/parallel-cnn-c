#ifndef LAYERS_H
#define LAYERS_H

#define INPUT_SIZE 6
#define FILTER_SIZE 3
#define POOL_SIZE 2
#define FC_INPUT_SIZE 16
#define FC_OUTPUT_SIZE 1

typedef struct {
    float *data;  // size = INPUT_SIZE * INPUT_SIZE
} Input;

typedef struct {
    float *weights;  // size = FILTER_SIZE * FILTER_SIZE
    float bias;
} ConvLayer;

typedef struct {
    float *data;  // size = CONV_SIZE * CONV_SIZE
} ConvOutput;

typedef struct {
    float *data;  // size = POOL_SIZE * POOL_SIZE
} PoolOutput;

typedef struct {
    float *data;  // size = FC_INPUT_SIZE
} FlattenOutput;

typedef struct {
    float *weights; // size = FC_OUTPUT_SIZE * FC_INPUT_SIZE
    float *biases;  // size = FC_OUTPUT_SIZE
} FullyConnectedLayer;

void conv_forward(Input *input, ConvLayer *layer, ConvOutput *output);
void maxpool_forward(ConvOutput *input, PoolOutput *output);
void flatten_forward(PoolOutput *input, FlattenOutput *output);
void fc_forward(FlattenOutput *input, FullyConnectedLayer *fc, float *output);

#endif
