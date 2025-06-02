#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cnn.h"
#include "utils.h"
#include "cnn_cuda.h"

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__device__ float leaky_relu_cuda(float x) {
    return x > 0 ? x : 0.01f * x;
}

__device__ float relu_cuda(float x) {
    return x > 0 ? x : 0;
}

__global__ void conv_forward_kernel(const float *input, const float *weights, const float *biases,
                                    float *output, int in_width, int in_height, int in_channel,
                                    int out_width, int out_height, int out_channel, int kernel_size) {
    int out_c = blockIdx.z;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < out_width && out_y < out_height && out_c < out_channel) {
        float sum = 0.0f;
        for (int in_c = 0; in_c < in_channel; ++in_c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_x = out_x + kx;
                    int in_y = out_y + ky;
                    int in_idx = in_c * in_height * in_width + in_y * in_width + in_x;
                    int w_idx = out_c * in_channel * kernel_size * kernel_size + in_c * kernel_size * kernel_size + ky * kernel_size + kx;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
        int out_idx = out_c * out_height * out_width + out_y * out_width + out_x;
        output[out_idx] = relu_cuda(sum + biases[out_c]);
    }
}

Tensor3D conv_forward_cuda(Tensor3D input, ConvLayer *layer) {
    int out_w = input.width - layer->kernel_size + 1;
    int out_h = input.height - layer->kernel_size + 1;
    int out_c = layer->out_channels;
    int in_c = layer->in_channels;
    int ks = layer->kernel_size;
    int input_size = input.width * input.height * input.channels;
    int output_size = out_w * out_h * out_c;
    int weight_size = out_c * in_c * ks * ks;

    float *d_input, *d_weights, *d_biases, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * input_size));
    CUDA_CHECK(cudaMalloc(&d_weights, sizeof(float) * weight_size));
    CUDA_CHECK(cudaMalloc(&d_biases, sizeof(float) * out_c));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * output_size));

    CUDA_CHECK(cudaMemcpy(d_input, input.data, sizeof(float) * input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, layer->weights, sizeof(float) * weight_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases, layer->biases, sizeof(float) * out_c, cudaMemcpyHostToDevice));

    dim3 threads(16, 16);
    dim3 blocks((out_w + 15) / 16, (out_h + 15) / 16, out_c);
    conv_forward_kernel<<<blocks, threads>>>(d_input, d_weights, d_biases, d_output,
                                             input.width, input.height, in_c,
                                             out_w, out_h, out_c, ks);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *output_data = (float*)malloc(sizeof(float) * output_size);
    CUDA_CHECK(cudaMemcpy(output_data, d_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);

    return (Tensor3D){ out_w, out_h, out_c, output_data };
}

__global__ void maxpool_forward_kernel(const float *input, float *output,
                                       int in_w, int in_h, int in_c,
                                       int pool_size, int out_w, int out_h) {
    int c = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < out_h && j < out_w && c < in_c) {
        float max_val = -1e9;
        for (int pi = 0; pi < pool_size; ++pi) {
            for (int pj = 0; pj < pool_size; ++pj) {
                int y = i * pool_size + pi;
                int x = j * pool_size + pj;
                int idx = c * in_h * in_w + y * in_w + x;
                max_val = fmaxf(max_val, input[idx]);
            }
        }
        int out_idx = c * out_h * out_w + i * out_w + j;
        output[out_idx] = max_val;
    }
}

Tensor3D maxpool_forward_cuda(Tensor3D input, int pool_size) {
    int in_w = input.width;
    int in_h = input.height;
    int in_c = input.channels;
    int out_w = in_w / pool_size;
    int out_h = in_h / pool_size;

    float *d_input, *d_output;
    size_t in_size = in_w * in_h * in_c * sizeof(float);
    size_t out_size = out_w * out_h * in_c * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input, in_size));
    CUDA_CHECK(cudaMemcpy(d_input, input.data, in_size, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_output, out_size));

    dim3 blockSize(16, 16);
    dim3 gridSize((out_w + 15) / 16, (out_h + 15) / 16, in_c);
    maxpool_forward_kernel<<<gridSize, blockSize>>>(d_input, d_output, in_w, in_h, in_c, pool_size, out_w, out_h);

    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_output = (float *)malloc(out_size);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
    free(input.data);  // optional if no reuse

    return (Tensor3D){out_w, out_h, in_c, h_output};
}


Vector flatten_cuda(Tensor3D input) {
    int total = input.width * input.height * input.channels;
    float *flat = input.data; // Directly reuse
    return (Vector){ total, flat };
}

__global__ void fc_forward_kernel(const float *input, const float *weights, const float *biases,
                                  float *output, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_features) {
        float sum = 0.0f;
        for (int j = 0; j < in_features; j++) {
            sum += weights[idx * in_features + j] * input[j];
        }
        output[idx] = relu_cuda(sum + biases[idx]);
    }
}

Vector fc_forward_cuda(Vector input, FullyConnectedLayer *layer) {
    int in_features = input.size;
    int out_features = layer->out_features;

    float *d_input, *d_weights, *d_biases, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(float) * in_features));
    CUDA_CHECK(cudaMalloc(&d_weights, sizeof(float) * in_features * out_features));
    CUDA_CHECK(cudaMalloc(&d_biases, sizeof(float) * out_features));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float) * out_features));

    CUDA_CHECK(cudaMemcpy(d_input, input.data, sizeof(float) * in_features, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, layer->weights, sizeof(float) * in_features * out_features, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_biases, layer->biases, sizeof(float) * out_features, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (out_features + threads - 1) / threads;
    fc_forward_kernel<<<blocks, threads>>>(d_input, d_weights, d_biases, d_output,
                                           in_features, out_features);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *output_data = (float*)malloc(sizeof(float) * out_features);
    CUDA_CHECK(cudaMemcpy(output_data, d_output, sizeof(float) * out_features, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_output);

    free(input.data);
    return (Vector){ out_features, output_data };
}

void cnn_forward_cuda(CNN *cnn) {
    int total_input = cnn->input_width * cnn->input_height * cnn->input_channels;
    float *copy = (float*)malloc(sizeof(float) * total_input);
    for (int i = 0; i < total_input; i++) copy[i] = cnn->input_data[i];

    Tensor3D x = { cnn->input_width, cnn->input_height, cnn->input_channels, copy };

    for (int i = 0; i < cnn->num_conv_layers; i++) {
        x = conv_forward_cuda(x, &cnn->conv_layers[i]);
        // x = maxpool_forward_cuda(x, 2);
    }

    Vector v = flatten_cuda(x);
    for (int i = 0; i < cnn->num_fc_layers; i++) {
        // int apply_activation = (i != cnn->num_fc_layers - 1);
        v = fc_forward_cuda(v, &cnn->fc_layers[i]);
    }
    cnn->output = v.data[0];

    free(v.data);
}