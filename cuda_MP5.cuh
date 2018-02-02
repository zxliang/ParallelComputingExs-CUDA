#ifndef CUDA_MP5_CUH
#define CUDA_MP5_CUH

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

#define NUM_ELEMENTS 512
#define RD_BLOCK_SIZE 64

int cuda_MP5(int argc, char* argv[]);
int ReadFileData_MP5(float* M, char* file_name);
void computeGold_MP5(float* reference, float* idata, const unsigned int len);
float computeOnDevice_MP5(float* h_data, int num_elements);
__global__ void reduction_kernel_MP5(float *g_data, float *blocksum, int n);

#endif // !CUDA_MP5_CUH