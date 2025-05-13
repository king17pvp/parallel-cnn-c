#include "cnn.h"

void cnn_forward(CNN *cnn) {
    conv_forward(&cnn->input, &cnn->conv, &cnn->conv_out);
    maxpool_forward(&cnn->conv_out, &cnn->pool_out);
    flatten_forward(&cnn->pool_out, &cnn->flat_out);
    fc_forward(&cnn->flat_out, &cnn->fc, cnn->output);
}
