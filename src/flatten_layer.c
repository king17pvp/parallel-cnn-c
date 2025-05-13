#include "layers.h"

void flatten_forward(PoolOutput *input, FlattenOutput *output) {
    for (int i = 0; i < FC_INPUT_SIZE; i++) {
        output->data[i] = input->data[i];
    }
}
