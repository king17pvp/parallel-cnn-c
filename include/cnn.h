#ifndef CNN_H
#define CNN_H

#include "layers.h"

typedef struct {
    Input input;
    ConvLayer conv;
    ConvOutput conv_out;
    PoolOutput pool_out;
    FlattenOutput flat_out;
    FullyConnectedLayer fc;
    float output[FC_OUTPUT_SIZE];
} CNN;

void cnn_forward(CNN *cnn);

#endif
