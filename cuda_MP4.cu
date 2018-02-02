#include "cuda_MP4.cuh"

__constant__ float Mconst[KERNEL_SIZE][KERNEL_SIZE];

// Matrix convolution kernel specification
__global__ void ConvolutionKernel_MP4(Matrix N, Matrix P)
{
  int n = KERNEL_SIZE / 2;
  __shared__ float N_ds[BLOCK_SIZE][BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int row = blockIdx.y*blockDim.y + ty;
  int col = blockIdx.x*blockDim.x + tx;

  if (row < N.height && col < N.width)
	N_ds[ty][tx] = N.elements[row*N.width + col];
  else
	N_ds[ty][tx] = 0;

  __syncthreads();

  int This_tile_start_point_x = blockIdx.x * blockDim.x;
  int Next_tile_start_point_x = (blockIdx.x + 1) * blockDim.x;
  int This_tile_start_point_y = blockIdx.y * blockDim.y;
  int Next_tile_start_point_y = (blockIdx.y + 1) * blockDim.y;

  int N_start_point_x = col - n;
  int N_start_point_y = row - n;

  float Pvalue = 0.0;

  for (int i = 0; i < 5; i++) {
	for (int j = 0; j < 5; j++) {

	  int N_idx_x = N_start_point_x + j;
	  int N_idx_y = N_start_point_y + i;
	  if (N_idx_x >= 0 && N_idx_x < N.width && N_idx_y >= 0 && N_idx_y < N.height) {
		if ((N_idx_x >= This_tile_start_point_x) &&
		  (N_idx_x < Next_tile_start_point_x) &&
		  (N_idx_y >= This_tile_start_point_y) &&
		  (N_idx_y < Next_tile_start_point_y)) {
		  Pvalue += N_ds[ty + i - n][tx + j - n] * Mconst[i][j];
		}
		else {
		  Pvalue += N.elements[N_idx_y*N.width + N_idx_x] * Mconst[i][j];
		}
	  }

	}
  }

  if (row < P.height && col < P.width) {
	P.elements[row*P.width + col] = Pvalue;
  }
  

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int cuda_MP4(int argc, char* argv[]) {

  Matrix  M;
  Matrix  N;
  Matrix  P;

  srand(2012);

  if (argc != 5 && argc != 4)
  {
	// Allocate and initialize the matrices
	M = AllocateMatrix_MP4(KERNEL_SIZE, KERNEL_SIZE, 1);
	N = AllocateMatrix_MP4((rand() % 1024) + 1, (rand() % 1024) + 1, 1);
	P = AllocateMatrix_MP4(N.height, N.width, 0);
  }
  else
  {
	// Allocate and read in matrices from disk
	int* params = NULL;
	unsigned int data_read = 0;
//	cutReadFilei(argv[1], &params, &data_read, true);
	data_read = ReadFileDimension_MP4(params, argv[1]);

	if (data_read != 2) {
	  cout << "Error reading parameter file\n";
	  paramsFree(params);
	  return 1;
	}
	
	M = AllocateMatrix_MP4(KERNEL_SIZE, KERNEL_SIZE, 0);
	N = AllocateMatrix_MP4(params[0], params[1], 0);
	P = AllocateMatrix_MP4(params[0], params[1], 0);

	paramsFree(params);
//	(void)ReadFile(&M, argv[2]);
//	(void)ReadFile(&N, argv[3]);
	ReadFileData_MP4(&M, argv[2]);
	ReadFileData_MP4(&N, argv[3]);
  }

  cout << "pos3" << endl;

  // M * N on the device
  ConvolutionOnDevice(M, N, P);

  // compute the matrix convolution on the CPU for comparison
  Matrix reference = AllocateMatrix_MP4(P.height, P.width, 0);
  computeGold_MP4(reference.elements, M.elements, N.elements, N.height, N.width);

  // in this case check if the result is equivalent to the expected soluion
//  CUTBoolean res = cutComparefe(reference.elements, P.elements, P.width * P.height, 0.001f);
  bool res = compareGold_MP4(reference.elements, P.elements, P.width * P.height
	, 0.001f);
  printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

  if (argc == 5)
  {
	WriteFile_MP4(P, argv[4]);
  }
  else if (argc == 2)
  {
	WriteFile_MP4(P, argv[1]);
  }

  // Free matrices
  FreeMatrix_MP4(&M);
  FreeMatrix_MP4(&N);
  FreeMatrix_MP4(&P);

  return 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void ConvolutionOnDevice(const Matrix M, const Matrix N, Matrix P)
{
  // Load M and N to the device
  Matrix Md = AllocateDeviceMatrix_MP4(M);
  CopyToDeviceMatrix_MP4(Md, M);
  Matrix Nd = AllocateDeviceMatrix_MP4(N);
  CopyToDeviceMatrix_MP4(Nd, N);

  // Allocate P on the device
  Matrix Pd = AllocateDeviceMatrix_MP4(P);
  CopyToDeviceMatrix_MP4(Pd, P); // Clear memory

  // Setup the execution configuration
  dim3 dimGrid(ceil((float)P.width / BLOCK_SIZE), ceil((float)P.height / BLOCK_SIZE)
	, 1);

  cout << ceil((float)P.width / BLOCK_SIZE) << " " << ceil((float)P.height / BLOCK_SIZE) << endl;
  
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

  // Copy M to device constant memeroy Mconst
  cudaMemcpyToSymbol(Mconst, M.elements, KERNEL_SIZE*KERNEL_SIZE * sizeof(float));

  // Launch the device computation threads!
  ConvolutionKernel_MP4 <<< dimGrid, dimBlock >>>(Nd, Pd);

  cudaDeviceSynchronize();

  // Read P from the device
  CopyFromDeviceMatrix_MP4(P, Pd);

  // Free device matrices
  FreeDeviceMatrix_MP4(&Md);
  FreeDeviceMatrix_MP4(&Nd);
  FreeDeviceMatrix_MP4(&Pd);
}

// Allocate a device matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
//  If init == 2, initialize matrix parameters, but do not allocate memory
Matrix AllocateMatrix_MP4(int height, int width, int init)
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
	M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	if (rand() % 2)
	  M.elements[i] = -M.elements[i];
  }
  return M;
}

// Read dimension of matrix M and N from file
int ReadFileDimension_MP4(int* params, char* file_name)
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
int ReadFileData_MP4(Matrix* M, char* file_name)
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

void paramsFree(int* params)
{
  free(params);
  params = NULL;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A convolved with B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param kernel_size         height and width of matrix A
//! @param hB         height of matrices B and C
//! @param wB         width of matrices B and C
////////////////////////////////////////////////////////////////////////////////
void computeGold_MP4(float* C, const float* A, const float* B, unsigned int hB, 
  unsigned int wB)
{

  // For each element in the result matrix matrix
  for (unsigned int i = 0; i < hB; ++i) {
	for (unsigned int j = 0; j < wB; ++j) {
	  double sum = 0;
	  // check the start and end values of m and n to prevent overrunning the 
	  //  matrix edges
	  unsigned int mbegin = (i < 2) ? 2 - i : 0;
	  unsigned int mend = (i >(hB - 3)) ?
		hB - i + 2 : 5;
	  unsigned int nbegin = (j < 2) ? 2 - j : 0;
	  unsigned int nend = (j >(wB - 3)) ?
		(wB - j) + 2 : 5;
	  // overlay A over B centered at element (i,j).  For each 
	  //  overlapping element, multiply the two and accumulate
	  for (unsigned int m = mbegin; m < mend; ++m) {
		for (unsigned int n = nbegin; n < nend; n++) {
		  sum += A[m * 5 + n] *
			B[wB*(i + m - 2) + (j + n - 2)];
		}
	  }
	  // store the result
	  C[i*wB + j] = (float)sum;
	}
  }
}

bool compareGold_MP4(float* ref, const float* C, unsigned int N,
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

void WriteFile_MP4(Matrix M, char* file_name)
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
void FreeMatrix_MP4(Matrix* M)
{
  free(M->elements);
  M->elements = NULL;
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix_MP4(const Matrix M)
{
  Matrix Mdevice = M;
  int size = M.width * M.height * sizeof(float);
  cudaMalloc((void**)&Mdevice.elements, size);
  return Mdevice;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix_MP4(Matrix Mdevice, const Matrix Mhost)
{
  int size = Mhost.width * Mhost.height * sizeof(float);
  Mdevice.height = Mhost.height;
  Mdevice.width = Mhost.width;
  Mdevice.pitch = Mhost.pitch;
  cudaMemcpy(Mdevice.elements, Mhost.elements, size,
	cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix_MP4(Matrix Mhost, const Matrix Mdevice)
{
  int size = Mdevice.width * Mdevice.height * sizeof(float);
  cudaMemcpy(Mhost.elements, Mdevice.elements, size,
	cudaMemcpyDeviceToHost);
}

// Free a device matrix.
void FreeDeviceMatrix_MP4(Matrix* M)
{
  cudaFree(M->elements);
  M->elements = NULL;
}