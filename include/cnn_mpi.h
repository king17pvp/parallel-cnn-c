#ifndef CNN_MPI_H
#define CNN_MPI_H
#include "mpi.h"
#include "cnn.h"
void cnn_forward_mpi(CNN *cnn, MPI_Comm comm);
#endif