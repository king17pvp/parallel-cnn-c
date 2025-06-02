#ifndef CNN_HYBRID_H
#define CNN_HYBRID_H
#include "mpi.h"
#include "cnn.h"

void cnn_forward_hybrid(float *all_images, int total_images, CNN *cnn, float *all_results, MPI_Comm comm);
#endif
