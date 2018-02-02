#include "cuda_MP5.cuh"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int cuda_MP5(int argc, char* argv[])
{
  int num_elements = NUM_ELEMENTS;
  int errorM = 0;
  const unsigned int array_mem_size = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  float* h_data = (float*)malloc(array_mem_size);

  // * No arguments: Randomly generate input data and compare against the 
  //   host's result.
  // * One argument: Read the input data array from the given file.
  switch (argc - 1)
  {
	case 1:  // One Argument
	  // errorM = ReadFile(h_data, argv[1]);
	  errorM = ReadFileData_MP5(h_data, argv[1]);
	  if (errorM != 1)
	  {
		printf("Error reading input file!\n");
		exit(1);
	  }
	  break;

	default:  
	  // No Arguments or one argument
	  // initialize the input data on the host to be integer values
	  // between 0 and 1000
	  for (unsigned int i = 0; i < num_elements; ++i)
	  {
		h_data[i] = floorf(1000 * (rand() / (float)RAND_MAX));
	  }
	  break;
  }

  // compute reference solution
  float reference = 0.0f;
  computeGold_MP5(&reference, h_data, num_elements);

  // **===-------- Modify the body of this function -----------===**
  float result = computeOnDevice_MP5(h_data, num_elements);
  // **===-----------------------------------------------------------===**

  // We can use an epsilon of 0 since values are integral and in a range 
  // that can be exactly represented
  float epsilon = 0.0f;
  unsigned int result_regtest = (abs(result - reference) <= epsilon);
  printf("Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
  printf("device: %f  host: %f\n", result, reference);
  // cleanup memory
  free(h_data);

  return 0;
}

int ReadFileData_MP5(float* M, char* file_name)
{
  unsigned int data_read = NUM_ELEMENTS;
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

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void computeGold_MP5(float* reference, float* idata, const unsigned int len)
{
  reference[0] = 0;
  double total_sum = 0;
  unsigned int i;
  for (i = 0; i < len; ++i)
  {
	total_sum += idata[i];
  }
  *reference = total_sum;
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimensions, excutes kernel function, and copy result of scan back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
float computeOnDevice_MP5(float* h_data, int num_elements)
{

  float* d_data;
  cudaMalloc((void**)&d_data, sizeof(float)*num_elements);
  cudaMemcpy(d_data, h_data, sizeof(float)*num_elements,
	cudaMemcpyHostToDevice);

  int num_block = ceil((float)num_elements / (2 * RD_BLOCK_SIZE));
  float* d_block_sum;
  cudaMalloc((void**)&d_block_sum, sizeof(float)*num_block);

  // placeholder
  dim3 dimGrid, dimBlock;
  
  dimGrid.x = num_block;
  dimGrid.y = dimGrid.z = 1;
  dimBlock.x = RD_BLOCK_SIZE;
  dimBlock.y = dimBlock.z = 1;

  reduction_kernel_MP5 << <dimGrid, dimBlock >> > (d_data, d_block_sum, num_elements);

  cudaDeviceSynchronize();

  float* h_block_sum = (float*)malloc(sizeof(float)*num_block);
  cudaMemcpy(h_block_sum, d_block_sum, sizeof(float)*num_block,
	cudaMemcpyDeviceToHost);

  cudaMemcpy(h_data, d_data, sizeof(float)*num_elements,
	cudaMemcpyDeviceToHost);

  cudaFree(d_block_sum);
  cudaFree(d_data);

  float sum = 0;
  for (int i = 0; i < num_block; i++) {
	sum += h_block_sum[i];
//	cout << i << " " << h_block_sum[i] << endl;
//	cout << (i + 1) * 128 - 1 << " " << h_data[(i + 1) * 128 - 1] << endl;
  }

  return sum;
//  return h_data[num_elements-1];
}

// **===----------------- MP4.1 - Modify this function --------------------===**
//! @param g_idata  input data in global memory
//                  result is expected in index 0 of g_idata
//! @param n        input number of elements to scan from input data
// **===------------------------------------------------------------------===**
__global__ void reduction_kernel_MP5(float *g_data, float *blocksum, int n)
{
  __shared__ float ds_data[2 * RD_BLOCK_SIZE];
  unsigned int tx = threadIdx.x;
  unsigned int id = threadIdx.x + blockIdx.x*blockDim.x;
  if (2 * id + 1 < n) {
	ds_data[2 * tx] = g_data[2 * id];
	ds_data[2 * tx + 1] = g_data[2 * id + 1];
  }

  for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
	__syncthreads();
	unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
	if (index < 2 * blockDim.x)
	  ds_data[index] += ds_data[index - stride];
  }
  __syncthreads();

  if (threadIdx.x == 0)
	blocksum[blockIdx.x] = ds_data[2 * blockDim.x - 1];


  if (2 * id + 1 < n) {
	g_data[2 * id] = ds_data[2 * tx];
	g_data[2 * id + 1] = ds_data[2 * tx + 1];
  }
  
}