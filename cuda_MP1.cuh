#ifndef CUDA_MP1_CUH
#define CUDA_MP1_CUH

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <fstream>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_MP1_vectoradd.h"

using namespace std;

__global__ void VectorAddKernel(Vector A, Vector B, Vector C);

int cuda_MP1(int argc, char* argv[]);
Vector AllocateVector(int size, int init);
int ReadFile(Vector* V, char* file_name);
void VectorAddOnDevice(const Vector A, const Vector B, Vector C);
Vector AllocateDeviceVector(const Vector V);
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost);
void computeGold(float* C, const float* A, const float* B, unsigned int N);
bool compareGold(float* ref, const float* C, unsigned int N, float precision);
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice);
void WriteFile(Vector V, char* file_name);

#endif // !CUDA_MP1_CUH
