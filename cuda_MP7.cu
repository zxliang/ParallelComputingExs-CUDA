#include "cuda_MP7.cuh"

void cuda_MP7(int argc, char* argv[])
{
  /* Case of 0 arguments: Default seed is used */
  if (argc < 2) {
	srand(0);
  }
  /* Case of 1 argument: Seed is specified as first command line argument */
  else {
	int seed = atoi(argv[1]);
	srand(seed);
  }

  uint8_t *gold_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH * sizeof(uint8_t));

  // Use kernel_bins for your final result
  uint8_t *kernel_bins = (uint8_t*)malloc(HISTO_HEIGHT*HISTO_WIDTH * sizeof(uint8_t));

  // A 2D array of histogram bin-ids.  One can think of each of these bins-ids as
  // being associated with a pixel in a 2D image.
  uint32_t **input = generate_histogram_bins();

  cout << "Input example: " << endl;
  for (int i = 0; i < 10; i++) {
	for (int j = 0; j < 14; j++) {
	  cout << input[i][j] << " ";
	}
	cout << endl;
  }

//  TIME_IT("ref_2dhisto",
//	  50,
//	  ref_2dhisto(input, INPUT_HEIGHT, INPUT_WIDTH, gold_bins);)

  ref_2dhisto(input, INPUT_HEIGHT, INPUT_WIDTH, gold_bins);
	/* Include your setup code below (temp variables, function calls, etc.) */

  uint32_t **d_input = NULL;
  cudaMalloc((void**)&d_input, INPUT_HEIGHT * INPUT_WIDTH * sizeof(uint32_t));

  uint8_t *d_obins = NULL;
  cudaMalloc((void**)&d_obins, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint8_t));
  
  uint32_t *temp_bins = NULL;
  cudaMalloc((void**)&temp_bins, HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t));

  cudaMemcpy(d_input, &(input[0][0]), INPUT_HEIGHT * INPUT_WIDTH * sizeof(uint32_t),
	cudaMemcpyHostToDevice);

	/* End of setup code */

	/* This is the call you will use to time your parallel implementation */
//	TIME_IT("opt_2dhisto",
//	  50,
//	  opt_2dhisto( /*Define your own function parameters*/);)

  opt_2dhisto(d_input, d_obins, temp_bins, INPUT_HEIGHT, INPUT_WIDTH);
	/* Include your teardown code below (temporary variables, function calls, etc.) */

  cudaMemcpy(kernel_bins, d_obins, HISTO_HEIGHT*HISTO_WIDTH * sizeof(uint8_t),
	cudaMemcpyDeviceToHost);

  cudaFree(temp_bins);
  cudaFree(d_input);
  cudaFree(d_obins);
	/* End of teardown code */

  int passed = 1;
  cout << "Gold_Bins vs. Kernal_Bins" << endl;
  for (int i = 0; i < HISTO_HEIGHT*HISTO_WIDTH; i++) {
	if (gold_bins[i] != kernel_bins[i]) {
	  cout << i << " " << gold_bins[i] << " " << kernel_bins[i] << endl;
	  passed = 0;
	  break;
	}
  }
  (passed) ? printf("\n    Test PASSED\n") : printf("\n    Test FAILED\n");

  free(gold_bins);
  free(kernel_bins);
}

int ref_2dhisto(uint32_t *input[], size_t height, size_t width, uint8_t bins[])
{

  // Zero out all the bins
  memset(bins, 0, HISTO_HEIGHT*HISTO_WIDTH * sizeof(bins[0]));

  for (size_t j = 0; j < height; ++j)
  {
	for (size_t i = 0; i < width; ++i)
	{
	  const uint32_t value = input[j][i];

	  uint8_t *p = (uint8_t*)bins;

	  // Increment the appropriate bin, but do not roll-over the max value
	  if (p[value] < UINT8_MAX)
		++p[value];
	}
  }

  return 0;
}

void** alloc_2d(size_t y_size, size_t x_size, size_t element_size)
{
  const size_t x_size_padded = (x_size + 128) & 0xFFFFFF80;

  uint8_t *data = (uint8_t*)calloc(x_size_padded * y_size, element_size);
  void   **res = (void**)calloc(y_size, sizeof(void*));

  if (data == 0 || res == 0)
  {
	free(data);
	free(res);
	res = 0;
	goto exit;
  }

  for (size_t i = 0; i < y_size; ++i)
	res[i] = data + (i * x_size_padded * element_size);

exit:
  return res;
}

// Generate another bin for the histogram.  The bins are created as a random walk ...
static uint32_t next_bin(uint32_t pix)
{
  const uint16_t bottom = pix & ((1 << HISTO_LOG) - 1);
  const uint16_t top = (uint16_t)(pix >> HISTO_LOG);

  int new_bottom = NEXT(bottom, SPREAD_BOTTOM)
	CLAMP(new_bottom, 0, HISTO_WIDTH - 1)

  int new_top = NEXT(top, SPREAD_TOP)
	CLAMP(new_top, 0, HISTO_HEIGHT - 1)

	const uint32_t result = (new_bottom | (new_top << HISTO_LOG));

  return result;
}

// Return a 2D array of histogram bin-ids.  This function generates
// bin-ids with correlation characteristics similar to some actual images.
// The key point here is that the pixels (and thus the bin-ids) are *NOT*
// randomly distributed ... a given pixel tends to be similar to the
// pixels near it.
static uint32_t **generate_histogram_bins()
{
  uint32_t **input = (uint32_t**)alloc_2d(INPUT_HEIGHT, INPUT_WIDTH, sizeof(uint32_t));

  input[0][0] = HISTO_WIDTH / 2 | ((HISTO_HEIGHT / 2) << HISTO_LOG);
  for (int i = 1; i < INPUT_WIDTH; ++i)
	input[0][i] = next_bin(input[0][i - 1]);
  for (int j = 1; j < INPUT_HEIGHT; ++j)
  {
	input[j][0] = next_bin(input[j - 1][0]);
	for (int i = 1; i < INPUT_WIDTH; ++i)
	  input[j][i] = next_bin(input[j][i - 1]);
  }

  return input;
}

/*
int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
  // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
  // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
  // until 00:00:00 January 1, 1970 
  static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

  SYSTEMTIME  system_time;
  FILETIME    file_time;
  uint64_t    time;

  GetSystemTime(&system_time);
  SystemTimeToFileTime(&system_time, &file_time);
  time = ((uint64_t)file_time.dwLowDateTime);
  time += ((uint64_t)file_time.dwHighDateTime) << 32;

  tp->tv_sec = (long)((time - EPOCH) / 10000000L);
  tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
  return 0;
}
*/

void opt_2dhisto(uint32_t **d_input, uint8_t *d_bins, uint32_t *d_temp_bins, 
  size_t height, size_t width)
{
  /* This function should only contain a call to the GPU
  histogramming kernel. Any memory allocations and
  transfers must be done outside this function */

  unsigned int num_blocks = ceil((float)width / TILE_SIZE_MP7);

  histo_kernel << <num_blocks, 1024 >> > (d_input, d_temp_bins);

  histo_32to8_kernel << <1, 1024 >> >(d_bins, d_temp_bins, 1024);

  cudaThreadSynchronize();

}

__global__ void histo_kernel(uint32_t **d_input, uint32_t *d_ouput)
{
  __shared__ uint32_t private_bins[HISTO_HEIGHT];
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int ti = threadIdx.x;
  private_bins[ti] = 0;

  __syncthreads();

  int start_col = TILE_SIZE_MP7 * blockIdx.x;

  for (int i = 0; i < TILE_SIZE_MP7; i++) {
	if (start_col + i < INPUT_WIDTH)
	  atomicAdd(&private_bins[d_input[ti][start_col + i]], 1);
  }

  __syncthreads();

  atomicAdd(&(d_ouput[ti]), private_bins[ti]);
  atomicAdd(&(d_ouput[ti + 512]), private_bins[ti + 512]);
}

__global__ void histo_cuda_kernel(int *d_input, int *d_output)
{
  __shared__ int private_bins[HISTO_MAX];
//  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int ti = threadIdx.x;

  if (ti < HISTO_MAX)
	private_bins[ti] = 0;

  __syncthreads();

  int start_col = TILE_SIZE_MP7 * blockIdx.x;

  for (int i = 0; i < TILE_SIZE_MP7; i++) {
	if (start_col + i < INPUT_WIDTH)
	  atomicAdd(&(private_bins[d_input[ti * INPUT_WIDTH + (start_col + i)]]), 1);
  }

  __syncthreads();

  if (ti < HISTO_MAX)
	atomicAdd(&(d_output[ti]), private_bins[ti]);

}


__global__ void histo_32to8_kernel(uint8_t *d_ouput, uint32_t *d_temp, const int sz)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < sz)
  {
	if (d_temp[idx]< UINT8_MAX)
	  d_ouput[idx] = (uint8_t)d_temp[idx];
	if (d_temp[idx] >= UINT8_MAX)
	  d_ouput[idx] = (uint8_t)UINT_MAX;
  }
  __syncthreads();
}


void histo_cuda()
{
  cout << "Starting CUDA histo test..." << endl;

  int *input_array = (int*)malloc(INPUT_HEIGHT * INPUT_WIDTH * sizeof(int));;
  int gold_histo[HISTO_MAX] = { 0 };
  int kernel_histo[HISTO_MAX] = { 0 };

  for (int i = 0; i < INPUT_HEIGHT; i++)
	for (int j = 0; j < INPUT_WIDTH; j++)
	  input_array[i * INPUT_WIDTH+ j] = rand() % (HISTO_MAX - 1);

  for (int i = 0; i < 10; i++) {
	for (int j = 0; j < 14; j++) {
	  cout << input_array[i * INPUT_WIDTH + j] << " ";
	}
	cout << endl;
  }

  for (int i = 0; i < INPUT_HEIGHT; i++)
	for (int j = 0; j < INPUT_WIDTH; j++)
	  gold_histo[input_array[i * INPUT_WIDTH + j]]++;

  cout << "Golden result..." << endl;
  for (int i = 0; i < 20; i++)
	cout << gold_histo[i] << " ";
  cout << endl;

  int *d_input = NULL;
  cudaMalloc((void**)&d_input, INPUT_HEIGHT * INPUT_WIDTH * sizeof(int));

  cudaMemcpy(d_input, input_array, INPUT_HEIGHT * INPUT_WIDTH * 
	sizeof(int), cudaMemcpyHostToDevice);

  int *d_output = NULL;
  cudaMalloc((void**)&d_output, HISTO_MAX * sizeof(int));

  unsigned int num_blocks = ceil((float)INPUT_WIDTH / TILE_SIZE_MP7);

  histo_cuda_kernel << <num_blocks, 1024 >> > (d_input, d_output);

  cudaThreadSynchronize();

  cudaMemcpy(kernel_histo, d_output, HISTO_MAX * sizeof(int),
	cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  cout << "Kernel result..." << endl;
  for (int i = 0; i < 20; i++)
	cout << kernel_histo[i] << " ";
  cout << endl;

  bool passed = true;
  cout << "Gold_Bins vs. Kernal_Bins" << endl;
  for (int i = 0; i < HISTO_MAX; i++) {
	if (gold_histo[i] != kernel_histo[i]) {
	  cout << i << ": " << gold_histo[i] << " vs. " << kernel_histo[i] << endl;
	  passed = false;
	  break;
	}
  }
  (passed) ? printf("\n    Test PASSED\n") : printf("\n    Test FAILED\n");

  cout << "Ending CUDA histo test..." << endl;
}