#ifndef CNN_CUDA_H
#define CNN_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cnn.h"

Tensor3D conv_forward_cuda(Tensor3D input, ConvLayer *layer);
Tensor3D maxpool_forward_cuda(Tensor3D input, int pool_size);
Vector flatten_cuda(Tensor3D input);
Vector fc_forward_cuda(Vector input, FullyConnectedLayer *layer);
void cnn_forward_cuda(CNN *cnn);
#ifdef __cplusplus
}
#endif

#endif