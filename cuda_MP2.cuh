#ifndef CUDA_MP2_CUH
#define CUDA_MP2_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_MP2_matrixmul.h"

using namespace std;

__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P);

int cuda_MP2(int argc, char* argv[]);
Matrix AllocateMatrix(int height, int width, int init);
int ReadFile(Matrix* M, char* file_name);
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);
Matrix AllocateDeviceMatrix(const Matrix M);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
void computeGold(float* C, const float* A, const float* B, unsigned int hA,
  unsigned int wA, unsigned int wB);
void WriteFile(Matrix M, char* file_name);

#endif // !CUDA_MP2_CUH