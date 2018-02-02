#include "cuda_MP6.cuh"

////////////////////////////////////////////////////////////////////////////////
//! Run a scan test for CUDA
////////////////////////////////////////////////////////////////////////////////
int cuda_MP6(int argc, char* argv[])
{
  int errorM = 0;
  float device_time;
  float host_time;
  int* size = NULL; //(int*)malloc(1 * sizeof(int));
  unsigned int data2read = 1;
  int num_elements = 0; // Must support large, non-power-of-2 arrays

  // allocate host memory to store the input data
  unsigned int mem_size = sizeof(float) * num_elements;
  float* h_data = (float*)malloc(mem_size);

  // * No arguments: Randomly generate input data and compare against the 
  //   host's result.
  // * One argument: Randomly generate input data and write the result to
  //   file name specified by first argument
  // * Two arguments: Read the first argument which indicate the size of the array,
  //   randomly generate input data and write the input data
  //   to the second argument. (for generating random input data)
  // * Three arguments: Read the first file which indicate the size of the array,
  //   then input data from the file name specified by 2nd argument and write the
  //   SCAN output to file name specified by the 3rd argument.
  switch (argc - 1)
  {
  case 2:
	// Determine size of array
//	cutReadFilei(argv[1], &size, &data2read, true);
	data2read = ReadFileData_MP6(size, argv[1], 1);
	if (data2read != 1) {
	  printf("Error reading parameter file\n");
	  exit(1);
	}

	num_elements = size[0];

	// allocate host memory to store the input data
	mem_size = sizeof(float) * num_elements;
	h_data = (float*)malloc(mem_size);

	for (unsigned int i = 0; i < num_elements; ++i)
	{
	  h_data[i] = (int)(rand() % MAX_RAND);
	}
	WriteFile_MP6(h_data, argv[2], num_elements);
	break;

  case 3:  // Three Arguments
//	cutReadFilei(argv[1], &size, &data2read, true);
	data2read = ReadFileData_MP6(size, argv[1], 1);
	if (data2read != 1) {
	  printf("Error reading parameter file\n");
	  exit(1);
	}

	num_elements = size[0];

	// allocate host memory to store the input data
	mem_size = sizeof(float) * num_elements;
	h_data = (float*)malloc(mem_size);

//	errorM = ReadFile(h_data, argv[2], size[0]);
	errorM = ReadFileData_MP6(h_data, argv[2], size[0]);
	if (errorM != 1)
	{
	  printf("Error reading input file!\n");
	  exit(1);
	}
	break;

  default:  // No Arguments or one argument
			// initialize the input data on the host to be integer values
			// between 0 and 1000
			// Use DEFAULT_NUM_ELEMENTS num_elements
	num_elements = DEFAULT_NUM_ELEMENTS;

	// allocate host memory to store the input data
	mem_size = sizeof(float) * num_elements;
	h_data = (float*)malloc(mem_size);

	// initialize the input data on the host
	for (unsigned int i = 0; i < num_elements; ++i)
	{
	  //                h_data[i] = 1.0f;
	  h_data[i] = (int)(rand() % MAX_RAND);
	}
	break;
  }


  clock_t timer;
  
  // compute reference solution
  float* reference = (float*)malloc(mem_size);
  timer = clock();
  computeGold_MP6(reference, h_data, num_elements);
  printf("\n**===-------------------------------------------------===**\n");
  printf("Processing %d elements...\n", num_elements);
  host_time = clock() - timer;
  printf("Host CPU Processing time: %f (ms)\n", 1000*((float)host_time)
	/ CLOCKS_PER_SEC);

  // allocate device memory input and output arrays
  float* d_idata = NULL;
  float* d_odata = NULL;

  cudaMalloc((void**)&d_idata, mem_size);
  cudaMalloc((void**)&d_odata, mem_size);

  // copy host memory to device input array
  cudaMemcpy(d_idata, h_data, mem_size, cudaMemcpyHostToDevice);
  // initialize all the other device arrays to be safe
  cudaMemcpy(d_odata, h_data, mem_size, cudaMemcpyHostToDevice);

  // **===-------- MP4.2 - Allocate data structure here -----------===**
  // preallocBlockSums(num_elements);
  // **===-----------------------------------------------------------===**

  // Run just once to remove startup overhead for more accurate performance 
  // measurement
  prescanArray_v1(d_odata, d_idata, 16);

  // Run the prescan
  timer = clock();

  // **===-------- MP4.2 - Modify the body of this function -----------===**
  prescanArray_v1(d_odata, d_idata, num_elements);
  // **===-----------------------------------------------------------===**
  cudaThreadSynchronize();

  device_time = clock() - timer;
  printf("CUDA Processing time: %f (ms)\n", 1000 * ((float)device_time)
	/ CLOCKS_PER_SEC);
  printf("Speedup: %fX\n", device_time / host_time);

  // **===-------- MP4.2 - Deallocate data structure here -----------===**
  // deallocBlockSums();
  // **===-----------------------------------------------------------===**


  // copy result from device to host
  cudaMemcpy(h_data, d_odata, sizeof(float) * num_elements,	cudaMemcpyDeviceToHost);

  if ((argc - 1) == 3)  // Three Arguments, write result to file
  {
	WriteFile_MP6(h_data, argv[3], num_elements);
  }
  else if ((argc - 1) == 1)  // One Argument, write result to file
  {
	WriteFile_MP6(h_data, argv[1], num_elements);
  }
/*
  cout << "i, h_data, ref, cuda calculations: \n";
  for (int i = 0; i < 514; i++) {
	cout << i << " " << reference[i] << " " << h_data[i] << endl;
  }
*/

  // Check if the result is equivalent to the expected soluion
//  unsigned int result_regtest = cutComparef(reference, h_data, num_elements);
  bool result_regtest = compareGold_MP6(reference, h_data, num_elements);
  printf("Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");

  // cleanup memory
  free(h_data);
  free(reference);
  cudaFree(d_odata);
  cudaFree(d_idata);

  return 0;
}

int ReadFileData_MP6(float* M, char* file_name, int size)
{
  unsigned int data_read = size;
  //  cutReadFilef(file_name, &(M->elements), &data_read, true);
  ifstream iFile(file_name);
  unsigned i = 0;
  if (iFile) {
	float data;
	while (iFile >> data) {
	  M[i++] = data;
	}
  }
  return (i != data_read);
}

int ReadFileData_MP6(int* M, char* file_name, int size)
{
  unsigned int data_read = size;
  //  cutReadFilef(file_name, &(M->elements), &data_read, true);
  ifstream iFile(file_name);
  unsigned i = 0;
  if (iFile) {
	int data;
	while (iFile >> data) {
	  M[i++] = data;
	}
  }
  return (i != data_read);
}

void WriteFile_MP6(float* M, char* file_name, int size)
{
  //  cutWriteFilef(file_name, M.elements, M.width*M.height, 0.0001f);
  ofstream oFile(file_name);
  if (oFile) {
	for (int i = 0; i < size; i++) {
	  oFile << M[i] << " ";
	}
	oFile.close();
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void computeGold_MP6(float* reference, float* idata, const unsigned int len)
{
  // inclusive scan
/*
  double total_sum = 0;
  for (unsigned int i = 0; i < len; ++i)
  {
	total_sum += idata[i];
	reference[i] = total_sum;
  }
*/
  // exclusive scan

  reference[0] = 0;
  double total_sum = 0;
  for (unsigned int i = 1; i < len; i++)
  {
	total_sum += idata[i - 1];
	reference[i] = idata[i - 1] + reference[i - 1];
  }
  if (total_sum != reference[len - 1])
	printf("Warning: exceeding single-precision accuracy.  Scan will be inaccurate.\n");


}

bool compareGold_MP6(float* ref, const float* C, const unsigned int N)
{
  double precision = 0.00001f;
  for (int i = 0; i < N; i++) {
	if (abs(ref[i] - C[i]) > precision) {
	  cout << i << ": " << ref[i] << ", " << C[i] << endl;
	  return false;
	}
  }
  return true;
}

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
// MP4.2 - You can use any other block size you wish.
#define BLOCK_SIZE_MP6 256

// MP4.2 - Host Helper Functions (allocate your own data structure...)


// MP4.2 - Device Functions


// MP4.2 - Kernel Functions


// **===-------- MP4.2 - Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray_v1(float *outArray, float *inArray, int numElements)
{
  int num_block = ceil((float)numElements / (2 * BLOCK_SIZE_MP6));

  dim3 dimGrid, dimBlock;

  dimGrid.x = num_block;
  dimGrid.y = dimGrid.z = 1;
  dimBlock.x = BLOCK_SIZE_MP6;
  dimBlock.y = dimBlock.z = 1;

  float* sumArray = NULL;
  cudaMalloc((void**)&sumArray, sizeof(float) * num_block);

  reduction_kernel_MP6 << <dimGrid, dimBlock >> > (outArray, inArray, sumArray, numElements);
  cudaDeviceSynchronize();

  dimGrid.x = 1;
  post_scan_kernel_MP6 << <dimGrid, dimBlock >> > (sumArray, num_block);
  cudaDeviceSynchronize();

  dimGrid.x = num_block;
  post_process_kernel_MP6 << <dimGrid, dimBlock >> > (outArray, sumArray, numElements);
  cudaDeviceSynchronize();

  cudaFree(sumArray);
}
// **===-----------------------------------------------------------===**

__global__ void reduction_kernel_MP6(float *out_data, float *in_data, 
  float *sum_data, int n)
{
  __shared__ float ds_data[2 * BLOCK_SIZE_MP6];
  unsigned int tx = threadIdx.x;
  unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

  ds_data[2 * tx] = (2 * id < n ? in_data[2 * id] : 0);
  ds_data[2 * tx + 1] = (2 * id + 1 < n ? in_data[2 * id + 1] : 0);

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
	__syncthreads();
	unsigned int index = (tx + 1) * 2 * stride - 1;
	if (index < 2 * blockDim.x)
	  ds_data[index] += ds_data[index - stride];
  }

  __syncthreads();
  if (threadIdx.x == 0)
	sum_data[blockIdx.x] = ds_data[2 * blockDim.x - 1];

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
	__syncthreads();
	unsigned int index = (tx + 1) * 2 * stride - 1;
	if (index + stride < 2 * blockDim.x)
	  ds_data[index + stride] += ds_data[index];
  }
  __syncthreads();

  if (2 * id < n) out_data[2 * id] = ds_data[2 * tx];
  if (2 * id + 1 < n) out_data[2 * id + 1] = ds_data[2 * tx + 1];

}

__global__ void post_scan_kernel_MP6(float *sumArray, int n)
{
  __shared__ float ds_data[2 * BLOCK_SIZE_MP6];
  unsigned int tx = threadIdx.x;
  unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

  ds_data[2 * tx] = (2 * id < n ? sumArray[2 * id] : 0);
  ds_data[2 * tx + 1] = (2 * id + 1 < n ? sumArray[2 * id + 1] : 0);

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
	__syncthreads();
	unsigned int index = (tx + 1) * 2 * stride - 1;
	if (index < 2 * blockDim.x)
	  ds_data[index] += ds_data[index - stride];
  }

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
	__syncthreads();
	unsigned int index = (tx + 1) * 2 * stride - 1;
	if (index + stride < 2 * blockDim.x)
	  ds_data[index + stride] += ds_data[index];
  }
  __syncthreads();

  if (2 * id < n) sumArray[2 * id] = ds_data[2 * tx];
  if (2 * id + 1 < n) sumArray[2 * id + 1] = ds_data[2 * tx + 1];
}


__global__ void post_process_kernel_MP6(float *outArray, float *sum_data, int n)
{
  unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (blockIdx.x > 0) {
	if (2 * id < n)
	  outArray[2 * id] += sum_data[blockIdx.x - 1];

	if (2 * id + 1 < n)
	  outArray[2 * id + 1] += sum_data[blockIdx.x - 1];
  }

}