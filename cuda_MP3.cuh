#ifndef CUDA_MP3_CUH
#define CUDA_MP3_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_MP2_matrixmul.h"

using namespace std;

// TILE size
#define TILE_SIZE 16

__global__ void MatrixMulKernel_MP3(Matrix M, Matrix N, Matrix P);

int cuda_MP3(int argc, char* argv[]);
Matrix AllocateMatrix_MP3(int height, int width, int init);
int ReadFileDimension_MP3(int* params, char* file_name);
int ReadFileData_MP3(Matrix* M, char* file_name);
void MatrixMulOnDevice_MP3(const Matrix M, const Matrix N, Matrix P);
void computeGold_MP3(float* C, const float* A, const float* B, unsigned int hA,
  unsigned int wA, unsigned int wB);
bool compareGold_MP3(float* ref, const float* C, unsigned int N, 
  float precision);
void WriteFile_MP3(Matrix M, char* file_name);
void FreeMatrix_MP3(Matrix* M);
Matrix AllocateDeviceMatrix_MP3(const Matrix M);
void CopyToDeviceMatrix_MP3(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix_MP3(Matrix Mhost, const Matrix Mdevice);
void FreeDeviceMatrix_MP3(Matrix* M);

#endif // !CUDA_MP3_CUH