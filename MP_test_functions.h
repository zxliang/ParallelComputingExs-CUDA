#ifndef MP_TEST_FUNCTIONS_H
#define MP_TEST_FUNCTIONS_H

#include "cuda_MP0.cuh"
#include "cuda_MP1.cuh"
#include "cuda_MP2.cuh"
#include "cuda_MP3.cuh"
#include "cuda_MP4.cuh"
#include "cuda_MP5.cuh"
#include "cuda_MP6.cuh"
#include "cuda_MP7.cuh"

int MP0_function_wrapper();
int MP1_function_wrapper(int argc, char* argv[]);
int MP2_function_wrapper(int argc, char* argv[]);
int MP3_function_wrapper(int argc, char* argv[]);
int MP4_function_wrapper(int argc, char* argv[]);
int MP5_function_wrapper(int argc, char* argv[]);
int MP6_function_wrapper(int argc, char* argv[]);
int MP7_function_wrapper(int argc, char* argv[]);

#endif // !MP_TEST_FUNCTIONS_H


