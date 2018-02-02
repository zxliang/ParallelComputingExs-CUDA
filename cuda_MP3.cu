#include "cuda_MP3.cuh"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel_MP3(Matrix M, Matrix N, Matrix P)
{

  __shared__ float M_ds[TILE_SIZE][TILE_SIZE];
  __shared__ float N_ds[TILE_SIZE][TILE_SIZE];

  unsigned ROW = threadIdx.y + blockDim.y * blockIdx.y;
  unsigned COL = threadIdx.x + blockDim.x * blockIdx.x;
  float Pvalue = 0.0;

  unsigned num_tiles = ceil((float)M.width / TILE_SIZE);

  for (int i = 0; i < num_tiles; i++) {
	if (ROW < M.height && (i*TILE_SIZE + threadIdx.x) < M.width)
	  M_ds[threadIdx.y][threadIdx.x] = M.elements[ROW*M.width + i*TILE_SIZE + 
	  threadIdx.x];
	else
	  M_ds[threadIdx.y][threadIdx.x] = 0.0;
	
	if (COL < N.width && (i*TILE_SIZE + threadIdx.y) < N.height)
	  N_ds[threadIdx.y][threadIdx.x] = N.elements[(i*TILE_SIZE + threadIdx.y)
	  *N.width + COL];
	else
	  N_ds[threadIdx.y][threadIdx.x] = 0.0;

	__syncthreads();

	for (int k = 0; k < TILE_SIZE; ++k) {
	  // if((i*16+k)<M.width && Row<M.height && (i*16+k)<N.height && Col<N.width)
	  Pvalue += M_ds[threadIdx.y][k] * N_ds[k][threadIdx.x];
	}
	__syncthreads();

  }

  if (ROW<P.height && COL<P.width)
	P.elements[ROW*P.width+COL] = Pvalue;

	  /*
  __shared__ float M_s[16][16];
  __shared__ float N_s[16][16];


  int bx = blockIdx.x;    int by = blockIdx.y;
  int tx = threadIdx.x;   int ty = threadIdx.y;

  int TILE_WIDTH = 16;

  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

  int num_block = ceil((float)M.width / TILE_WIDTH);

  for (int m = 0; m < num_block; ++m) {
	if (Row<M.height && (m*TILE_WIDTH + tx)<M.width)
	  M_s[ty][tx] = M.elements[Row*M.width + m*TILE_WIDTH + tx];
	else
	  M_s[ty][tx] = 0;

	if ((m * 16 + ty)<M.height && Col<N.width)
	  N_s[ty][tx] = N.elements[(m*TILE_WIDTH + ty)*N.width + Col];
	else
	  N_s[ty][tx] = 0;

	__syncthreads();

	for (int k = 0; k < TILE_WIDTH; ++k) {
	  // if((m*16+k)<M.width && Row<M.height && (m*16+k)<N.height && Col<N.width)
	  Pvalue += M_s[ty][k] * N_s[k][tx];
	}
	__syncthreads();
  }

  if (Row<P.height && Col<P.width)
	P.elements[Row*P.width + Col] = Pvalue;
	*/
}


int cuda_MP3(int argc, char* argv[])
{
  Matrix  M;
  Matrix  N;
  Matrix  P;
  int errorM = 0, errorN = 0;

  srand(52);

  if (argc != 5 && argc != 4)
  {
	// Allocate and initialize the matrices
	M = AllocateMatrix_MP3(rand() % 1024, rand() % 1024, 1);
	cout << "M: " << M.height << "x" << M.width << endl;
	N = AllocateMatrix_MP3(M.width, rand() % 1024, 1);
	cout << "N: " << N.height << "x" << N.width << endl;
	P = AllocateMatrix_MP3(M.height, N.width, 0);
  }
  else
  {
	// Allocate and read in matrices from disk
	int* params = NULL; //(int*)malloc(3 * sizeof(int));
	unsigned int data_read = 3;
//	cutReadFilei(argv[1], &params, &data_read, true);
	data_read = ReadFileDimension_MP3(params, argv[1]);
	if (data_read != 3) {
	  printf("Error reading parameter file\n");
	  return 1;
	}

	M = AllocateMatrix_MP3(params[0], params[1], 0);
	N = AllocateMatrix_MP3(params[1], params[2], 0);
	P = AllocateMatrix_MP3(params[0], params[2], 0);
	errorM = ReadFileData_MP3(&M, argv[2]);
	errorN = ReadFileData_MP3(&N, argv[3]);
	if (errorM || errorN)
	{
	  printf("Error reading input files %d, %d\n", errorM, errorN);
	  return 1;
	}
  }

  // M * N on the device
  MatrixMulOnDevice_MP3(M, N, P);

  printf("GPU computation complete\n");
  // compute the matrix multiplication on the CPU for comparison
  Matrix reference = AllocateMatrix_MP3(P.height, P.width, 0);
  computeGold_MP3(reference.elements, M.elements, N.elements, M.height, 
	M.width, N.width);

  printf("CPU computation complete\n");
  // in this case check if the result is equivalent to the expected soluion
  bool res = compareGold_MP3(reference.elements, P.elements, 
	P.height*P.width, 0.001f);
  printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

  if (argc == 5)
  {
	WriteFile_MP3(P, argv[4]);
  }
  else if (argc == 2)
  {
	WriteFile_MP3(P, argv[1]);
  }

  // Free matrices
  FreeMatrix_MP3(&M);
  FreeMatrix_MP3(&N);
  FreeMatrix_MP3(&P);

  return 0;
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory 
Matrix AllocateMatrix_MP3(int height, int width, int init)
{
  Matrix M;
  M.width = M.pitch = width;
  M.height = height;
  int size = M.width * M.height;
  M.elements = NULL;

  // don't allocate memory on option 2
  if (init == 2)
	return M;

  M.elements = (float*)malloc(size * sizeof(float));

  for (unsigned int i = 0; i < M.height * M.width; i++)
  {
	M.elements[i] = (init == 0) ? (0.0f) : (rand() * 3 / (float)RAND_MAX);
  }
  return M;
}

// Read dimension of matrix M and N from file
int ReadFileDimension_MP3(int* params, char* file_name)
{
  ifstream iFile(file_name);
  unsigned i = 0;
  if (iFile) {
	int data;
	while (iFile >> data) {
	  params[i++] = data;
	}
  }
  return i;
}

// Read a floating point matrix in from file
// Returns zero if the number of elements read is 
//  equals M.height * M.width, and 1 otherwise
int ReadFileData_MP3(Matrix* M, char* file_name)
{
  unsigned int data_read = M->height*M->width;
  //  cutReadFilef(file_name, &(M->elements), &data_read, true);
  ifstream iFile(file_name);
  unsigned i = 0;
  if (iFile) {
	float data;
	while (iFile >> data) {
	  M->elements[i++] = data;
	}
  }
  return (i != data_read);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice_MP3(const Matrix M, const Matrix N, Matrix P)
{
  // Load M and N to the device
  Matrix Md = AllocateDeviceMatrix_MP3(M);
  CopyToDeviceMatrix_MP3(Md, M);
  Matrix Nd = AllocateDeviceMatrix_MP3(N);
  CopyToDeviceMatrix_MP3(Nd, N);

  // Allocate P on the device
  Matrix Pd = AllocateDeviceMatrix_MP3(P);
  CopyToDeviceMatrix_MP3(Pd, P); // Clear memory

  // Setup the execution configuration
  dim3 dimGrid(ceil((float)P.width/TILE_SIZE), ceil((float)P.height/TILE_SIZE)
	, 1);
	
  dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

  // Launch the device computation threads!
  MatrixMulKernel_MP3 << <dimGrid, dimBlock >> >(Md, Nd, Pd);

  cudaDeviceSynchronize();

  // Read P from the device
  CopyFromDeviceMatrix_MP3(P, Pd);

  // Free device matrices
  FreeDeviceMatrix_MP3(&Md);
  FreeDeviceMatrix_MP3(&Nd);
  FreeDeviceMatrix_MP3(&Pd);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void computeGold_MP3(float* C, const float* A, const float* B, unsigned int hA, 
  unsigned int wA, unsigned int wB)
{
  for (unsigned int i = 0; i < hA; ++i)
	for (unsigned int j = 0; j < wB; ++j) {
	  double sum = 0;
	  for (unsigned int k = 0; k < wA; ++k) {
		double a = A[i * wA + k];
		double b = B[k * wB + j];
		sum += a * b;
	  }
	  C[i * wB + j] = (float)sum;
	}
}

bool compareGold_MP3(float* ref, const float* C, unsigned int N, 
  float precision)
{
  for (int i = 0; i < N; i++) {
	if (abs(ref[i] - C[i]) > precision) {
	  cout << i << ": " << ref[i] << ", " << C[i] << endl;
	  return false;
	}
  }
  return true;
}

void WriteFile_MP3(Matrix M, char* file_name)
{
  //  cutWriteFilef(file_name, M.elements, M.width*M.height, 0.0001f);
  ofstream oFile(file_name);
  if (oFile) {
	for (int i = 0; i < M.width*M.height; i++) {
	  oFile << M.elements[i] << " ";
	}
	oFile.close();
  }
}

// Free a host Matrix
void FreeMatrix_MP3(Matrix* M)
{
  free(M->elements);
  M->elements = NULL;
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix_MP3(const Matrix M)
{
  Matrix Mdevice = M;
  int size = M.width * M.height * sizeof(float);
  cudaMalloc((void**)&Mdevice.elements, size);
  return Mdevice;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix_MP3(Matrix Mdevice, const Matrix Mhost)
{
  int size = Mhost.width * Mhost.height * sizeof(float);
  Mdevice.height = Mhost.height;
  Mdevice.width = Mhost.width;
  Mdevice.pitch = Mhost.pitch;
  cudaMemcpy(Mdevice.elements, Mhost.elements, size,
	cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix_MP3(Matrix Mhost, const Matrix Mdevice)
{
  int size = Mdevice.width * Mdevice.height * sizeof(float);
  cudaMemcpy(Mhost.elements, Mdevice.elements, size,
	cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix_MP3(Matrix* M)
{
  cudaFree(M->elements);
  M->elements = NULL;
}