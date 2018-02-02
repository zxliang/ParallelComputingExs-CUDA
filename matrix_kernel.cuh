#ifndef MATRIX_KERNEL_H
#define MATRIX_KERNEL_H

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DIM 1900
#define BlockSize 32

#define TILE_SIZE 8
#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 8

__global__ void multi(int *A, int *B, int *C);
void matrixmulti(int A[][DIM], int B[][DIM], int C[][DIM]);

__global__ void addKernel(int *c, const int *a, const int *b);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
int cudaAdditionExample();

__global__ void Multiply_Kernel(int *A_d, int *B_d, int *C_h, unsigned int size);
cudaError_t MultiplyWithCuda(int *A_h, int *B_h, int *C_h, unsigned int size);
int ParallelMultiply(int *A_h, int *B_h, int *C_h, unsigned int size);

#endif // !MATRIX_KERNEL_H
