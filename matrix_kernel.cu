#include "matrix_kernel.cuh"

__global__ void multi(int *A, int *B, int *C)
{
  int cvalue = 0;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;


  if (row > DIM || col > DIM) return;

  for (int e = 0; e < DIM; ++e) {
	cvalue += A[row*DIM + e] * B[e*DIM + col];
  }
  C[row*DIM + col] = cvalue;
}

void matrixmulti(int A[][DIM], int B[][DIM], int C[][DIM]) {
  int *dev_a, *dev_b, *dev_c;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //allocate memory on global memory of gpu
  cudaError_t err = cudaMalloc((void**)&dev_a, ((DIM)*(DIM)) * sizeof(int));
  printf("Cuda malloc A:%s \n", cudaGetErrorString(err));
  err = cudaMalloc((void**)&dev_b, ((DIM)*(DIM)) * sizeof(int));
  printf("Cuda malloc B:%s \n", cudaGetErrorString(err));
  err = cudaMalloc((void**)&dev_c, ((DIM)*(DIM)) * sizeof(int));
  printf("Cuda malloc C:%s \n", cudaGetErrorString(err));


  //Copy array A and B on device allocated memory
  err = cudaMemcpy(dev_a, A, ((DIM*DIM)) * sizeof(int), cudaMemcpyHostToDevice);
  printf("Cuda memcpy to device A:%s \n", cudaGetErrorString(err));
  err = cudaMemcpy(dev_b, B, ((DIM*DIM)) * sizeof(int), cudaMemcpyHostToDevice);
  printf("Cuda memcpy to device B:%s \n", cudaGetErrorString(err));

  //two dimension threads
  dim3 dimBlock(BlockSize, BlockSize);
  dim3 dimGrid((DIM + dimBlock.x - 1) / dimBlock.x, (DIM + dimBlock.y - 1) / dimBlock.y);

  //call the kernel function multi
  cudaEventRecord(start);
  multi << < dimGrid, dimBlock >> >(dev_a, dev_b, dev_c);
  cudaEventRecord(stop);

  //retrieve array C from device memory
  err = cudaMemcpy(C, dev_c, ((DIM*DIM)) * sizeof(int), cudaMemcpyDeviceToHost);
  printf("Cuda memcpy to HOST C:%s \n", cudaGetErrorString(err));
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Elapsed time is %f ms\n", milliseconds);

  /*for (int i = 0; i < DIM; i++){
  for (int j = 0; j < DIM; j++){
  printf("C(%d,%d) = %d \n", i, j, C[i][j]);
  }
  }*/

  //free the memory
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}

int cudaAdditionExample()
{
  const int arraySize = 5;
  const int a[arraySize] = { 1, 2, 3, 4, 5 };
  const int b[arraySize] = { 10, 20, 30, 40, 50 };
  int c[arraySize] = { 0 };

  // Add vectors in parallel.
  cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "addWithCuda failed!");
	return 1;
  }

  printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	c[0], c[1], c[2], c[3], c[4]);

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceReset failed!");
	return 1;
  }

  return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
  int *dev_a = 0;
  int *dev_b = 0;
  int *dev_c = 0;
  cudaError_t cudaStatus;

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	goto Error;
  }

  // Allocate GPU buffers for three vectors (two input, one output)    .
  cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
  }

  cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
  }

  // Launch a kernel on the GPU with one thread for each element.
  addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	goto Error;
  }

  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	goto Error;
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
  }

Error:
  cudaFree(dev_c);
  cudaFree(dev_a);
  cudaFree(dev_b);

  return cudaStatus;
}

// my own multiply kernel
// one with shared memory

int ParallelMultiply(int *A_h, int *B_h, int *C_h, unsigned int size)
{
  cudaError_t cudaStatus = MultiplyWithCuda(A_h, B_h, C_h, size);
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "ParallelMultiply failed!");
	return 1;
  }

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
	fprintf(stderr, "cudaDeviceReset in ParallelMultiply failed!");
	return 1;
  }

  return 0;
}

__global__ void Multiply_Kernel(int *A_d, int *B_d, int *C_d, unsigned int size)
{
  __shared__ int A_ds[TILE_SIZE][TILE_SIZE];
  __shared__ int B_ds[TILE_SIZE][TILE_SIZE];

  int b_x = blockIdx.x, t_x = threadIdx.x;
  int b_y = blockIdx.y, t_y = threadIdx.y;
  int row = b_x * TILE_SIZE + t_x;
  int col = b_y * TILE_SIZE + t_y;

  int sum = 0;
  for (int m = 0; m < size / TILE_SIZE + 1; ++m) {
    if (row < size && m * TILE_SIZE + t_y < size)
	  A_ds[t_x][t_y] = A_d[row * size + m * TILE_SIZE + t_y];
	else
	  A_ds[t_x][t_y] = 0;

	if (col < size && m * TILE_SIZE + t_x < size)
	  B_ds[t_x][t_y] = B_d[(m * TILE_SIZE + t_x) * size + col];
	else
	  B_ds[t_x][t_y] = 0;

	__syncthreads();
	  for (int e = 0; e < TILE_SIZE; e++) {
		sum += A_ds[t_x][e] * B_ds[e][t_y];
		__syncthreads();
	  }
	}

	C_d[row*size + col] = sum;

}

cudaError_t MultiplyWithCuda(int *A_h, int *B_h, int *C_h, unsigned int size)
{
  int *A_d = 0;
  int *B_d = 0;
  int *C_d = 0;
  int matrix_size = size * size;
  cudaError_t status, status1, status2, status3;

  status = cudaSetDevice(0);
  if (status != cudaSuccess) {
	fprintf(stderr, "cudaSetDevice failed! A CUDA-capable GPU installed?");
	goto Error;
  }

  status1 = cudaMalloc((void**)&A_d, matrix_size * sizeof(int));
  status2 = cudaMalloc((void**)&B_d, matrix_size * sizeof(int));
  status3 = cudaMalloc((void**)&C_d, matrix_size * sizeof(int));

  if (status1 != cudaSuccess || status2 != cudaSuccess || status3 != cudaSuccess) {
	fprintf(stderr, "cudaMalloc failed!");
	goto Error;
  }

  status1 = cudaMemcpy(A_d, A_h, matrix_size * sizeof(int), cudaMemcpyHostToDevice);
  status2 = cudaMemcpy(B_d, B_h, matrix_size * sizeof(int), cudaMemcpyHostToDevice);

  if (status1 != cudaSuccess || status2 != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
  }

  dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
  int grid_x = size / BLOCK_SIZE_X + 1, grid_y = size / BLOCK_SIZE_Y + 1;
  dim3 dimGrid(grid_x, grid_y);

  Multiply_Kernel <<<dimGrid, dimBlock >>> (A_d, B_d, C_d, size);

  status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
	fprintf(stderr, "cudaDeviceSynchronize error code: %d !\n", status);
	goto Error;
  }

  // Copy output vector from GPU buffer to host memory.
  status3 = cudaMemcpy(C_h, C_d, matrix_size * sizeof(int), cudaMemcpyDeviceToHost);

  if (status3 != cudaSuccess) {
	fprintf(stderr, "cudaMemcpy failed!");
	goto Error;
  }

Error:
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  return status;
}

