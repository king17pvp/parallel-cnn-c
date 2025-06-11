# parallel-cnn-c

## Project Overview

`parallel-cnn-c` is a project that implements a Convolutional Neural Network (CNN) in C with parallel computing techniques. The goal is to leverage parallelism to optimize the performance of CNN operations, such as convolution, pooling, and activation functions.

## Features

- **Parallel Computing**: Utilizes multi-threading or other parallel computing techniques to speed up CNN operations.
- **Configurable Parameters**: Reads configuration values such as input dimensions, kernel size, and number of layers from a file.
- **Lightweight Implementation**: Written in C for high performance and low-level control.

## File Structure

- **src/**: Contains the source code files, including the main CNN implementation and utility functions.
- **build/**: Directory where compiled binaries are stored.
- **data/**: Directory for input data and configuration files.
- **README.md**: Documentation for the project.
- **Makefile**: Build instructions for the project.

## Prerequisites

Before running the project, ensure you have the following installed:

- GCC (GNU Compiler Collection)
- Make
- A compatible C runtime environment

## How to Run
1. Clone the repository:

   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:

   ```bash
   cd parallel-cnn-c
   ```
### 1) Serial implementation
Execute the commands in consoles as shown below:

   ```bash
   make
  ./build/cnn_exec
   ```

### 2) MPI implementation
Execute the commands in consoles as shown below:

   ```bash
   make -f MPI_Makefile
   ./build/cnn_exec_mpi
   ```

### 3) CUDA implementation
Execute the commands in consoles as shown below:

   ```bash
   make -f CUDA_Makefile
   ./build/cnn_exec_cuda
   ```

### 4) MPI + CUDA implementation
Execute the commands in consoles as shown below:

   ```bash
   make -f Hybrid_Makefile
   mpirun --allow-run-as-root -np 2 ./build/cnn_exec_hybrid load
   ```

## Configuration

The project reads configuration values from a file. The utility function in `src/utils.c` parses key-value pairs from the configuration file. Supported keys include:

- `input_width`: Width of the input image.
- `input_height`: Height of the input image.
- `input_channels`: Number of channels in the input image.
- `num_conv_layers`: Number of convolutional layers.
- `kernel_size`: Size of the convolutional kernel.
- `hidden_dim`: Dimension of the hidden layer.
- `mean`: Mean value for normalization.
- `std`: Standard deviation for normalization.
- `max_pool_stride`: Stride value for max pooling.

Ensure the configuration file is correctly formatted and placed in the appropriate directory.

## Debugging and Development

The `.gitignore` file excludes common build artifacts and temporary files, such as:

- Object files (`*.o`, `*.obj`)
- Executables (`*.exe`, `*.out`)
- Debug files (`*.pdb`, `*.dSYM`)
- IDE-specific files (`.vscode/`)

