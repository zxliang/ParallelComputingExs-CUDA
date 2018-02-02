#ifndef CUDA_MP4_CUH
#define CUDA_MP4_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#include "cuda_MP2_matrixmul.h"

// Thread block size
#define KERNEL_SIZE 5
#define BLOCK_SIZE 16

__global__ void ConvolutionKernel_MP4(Matrix N, Matrix P);

int cuda_MP4(int argc, char* argv[]);
Matrix AllocateMatrix_MP4(int height, int width, int init);
int ReadFileDimension_MP4(int* params, char* file_name);
int ReadFileData_MP4(Matrix* M, char* file_name);
void paramsFree(int* params);
void computeGold_MP4(float* C, const float* A, const float* B, unsigned int hB,
  unsigned int wB);
bool compareGold_MP4(float* ref, const float* C, unsigned int N,
  float precision);
void WriteFile_MP4(Matrix M, char* file_name);
void FreeMatrix_MP4(Matrix* M);
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P);
Matrix AllocateDeviceMatrix_MP4(const Matrix M);
void CopyToDeviceMatrix_MP4(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix_MP4(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix_MP4(Matrix* M);

#endif // !CUDA_MP4_CUH