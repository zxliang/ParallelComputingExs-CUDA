#ifndef CUDA_MP6_CUH
#define CUDA_MP6_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>

#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define DEFAULT_NUM_ELEMENTS 16000000
#define MAX_RAND 3

int cuda_MP6(int argc, char* argv[]);
int ReadFileData_MP6(float* M, char* file_name, int size);
int ReadFileData_MP6(int* M, char* file_name, int size);
void WriteFile_MP6(float* M, char* file_name, int size);
void computeGold_MP6(float* reference, float* idata, const unsigned int len);
void prescanArray_v1(float *outArray, float *inArray, int numElements);
bool compareGold_MP6(float* ref, const float* C, const unsigned int N);

__global__ void reduction_kernel_MP6(float *out_data, float *in_data,
  float *sum_data, int n);
__global__ void post_scan_kernel_MP6(float *sumArray, int n);
__global__ void post_process_kernel_MP6(float *outArray, float *sum_data, int n);

#endif // !CUDA_MP6_CUH