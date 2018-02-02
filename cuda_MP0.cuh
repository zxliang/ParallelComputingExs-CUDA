#ifndef CUDA_MP0_CUH
#define CUDA_MP0_CUH

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

const unsigned matrix_x = 4096;
const unsigned matrix_y = 4096;
const unsigned matrix_size = matrix_x * matrix_y;

__global__ void matrix_matrix_addition(float *C, const float * __restrict__ A,
  const float * __restrict__ B, const unsigned matrix_x, const unsigned matrix_y);

void FATAL(string msg);
int cuda_MP0();

#endif // !CUDA_MP0_CUH
