#ifndef CUDA_MP7_CUH
#define CUDA_MP7_CUH

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <time.h>
//#include <sys/time.h>
//#include <Windows.h>

#include <cuda.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

//#define __USE_BSD 1
#define INPUT_WIDTH  996
#define INPUT_HEIGHT 1024

#define HISTO_WIDTH  1024
#define HISTO_HEIGHT 1
#define HISTO_LOG 10

#define UINT8_MAX 255
#define HISTO_MAX 256

#define TILE_SIZE_MP7 8 

void cuda_MP7(int argc, char* argv[]);
int ref_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[]);
void** alloc_2d(size_t y_size, size_t x_size, size_t element_size);
static uint32_t next_bin(uint32_t pix);
//int gettimeofday(struct timeval * tp, struct timezone * tzp);
static uint32_t **generate_histogram_bins();
void opt_2dhisto(uint32_t **d_input, uint8_t *d_bins, uint32_t *d_temp_bins,
  size_t height, size_t width);

__global__ void histo_kernel(uint32_t **d_input, uint32_t *d_ouput);
__global__ void histo_cuda_kernel(int *d_input, int *d_output);
__global__ void histo_32to8_kernel(uint8_t *d_ouput, uint32_t *d_temp, const int sz);

void histo_cuda();

/*
#define TIME_IT(ROUTINE_NAME__, LOOPS__, ACTION__)\
{\
    printf("    Timing '%s' started\n", ROUTINE_NAME__);\
    struct timeval tv;\
    struct timezone tz;\
    const clock_t startTime = clock();\
    gettimeofday(&tv, &tz); long GTODStartTime =  tv.tv_sec * 1000 + tv.tv_usec / 1000 ;\
    for (int loops = 0; loops < (LOOPS__); ++loops)\
    {\
        ACTION__;\
    }\
    gettimeofday(&tv, &tz); long GTODEndTime =  tv.tv_sec * 1000 + tv.tv_usec / 1000 ;\
    const clock_t endTime = clock();\
    const clock_t elapsedTime = endTime - startTime;\
    const double timeInSeconds = (elapsedTime/(double)CLOCKS_PER_SEC);\
    printf("        GetTimeOfDay Time (for %d iterations) = %g\n", LOOPS__, (double)(GTODEndTime - GTODStartTime) / 1000. );\
    printf("        Clock Time        (for %d iterations) = %g\n", LOOPS__, timeInSeconds );\
    printf("    Timing '%s' ended\n", ROUTINE_NAME__);\
}
*/

#define SQRT_2    1.4142135623730950488
#define SPREAD_BOTTOM   (2)
#define SPREAD_TOP      (6)

#define NEXT(init_, spread_)\
    (init_ + (int)((rand() - 0.5) * (rand() - 0.5) * 4.0 * SQRT_2 * SQRT_2 * spread_));

#define CLAMP(value_, min_, max_)\
    if (value_ < 0)\
        value_ = (min_);\
    else if (value_ > (max_))\
        value_ = (max_);


#endif // !CUDA_MP7_CUH