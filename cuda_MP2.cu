#include "cuda_MP2.cuh"
#include "cuda_MP1.cuh"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
  //Multiply the two matrices
  unsigned row = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned col = threadIdx.y + blockDim.y * blockIdx.y;

  if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
	float sum = 0.0;
	for (int i = 0; i < MATRIX_SIZE; i++) {
	  sum += M.elements[row*MATRIX_SIZE+i] * N.elements[i*MATRIX_SIZE+col];
	}
	P.elements[row*MATRIX_SIZE + col] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int cuda_MP2(int argc, char** argv) {

  // Matrices for the program
  Matrix  M;
  Matrix  N;
  Matrix  P;
  // Number of elements in the solution matrix
  // Assuming square matrices, so the sizes of M, N and P are equal
  unsigned int size_elements = WP * HP;
  int errorM = 0, errorN = 0;

  srand(2012);

  // Check command line for input matrix files
  if (argc != 3 && argc != 4)
  {
	// No inputs provided
	// Allocate and initialize the matrices
	M = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	N = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
	P = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
  }
  else
  {
	// Inputs provided
	// Allocate and read source matrices from disk
	M = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	N = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	P = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	errorM = ReadFile(&M, argv[1]);
	errorN = ReadFile(&N, argv[2]);
	// check for read errors
	if (errorM != size_elements || errorN != size_elements)
	{
	  printf("Error reading input files %d, %d\n", errorM, errorN);
	  return 1;
	}
  }

  // M * N on the device
  MatrixMulOnDevice(M, N, P);

  // compute the matrix multiplication on the CPU for comparison
  Matrix reference = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
  computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);

  // check if the device result is equivalent to the expected solution
  bool res = compareGold(reference.elements, P.elements, size_elements, 0.0001f);
  printf("Test %s\n", (true == res) ? "PASSED" : "FAILED");

  // output result if output file is requested
  if (argc == 4)
  {
	WriteFile(P, argv[3]);
  }
  else if (argc == 2)
  {
	WriteFile(P, argv[1]);
  }

  // Free host matrices
  free(M.elements);
  M.elements = NULL;
  free(N.elements);
  N.elements = NULL;
  free(P.elements);
  P.elements = NULL;

  return 0;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix AllocateMatrix(int height, int width, int init)
{
  Matrix M;
  M.width = M.pitch = width;
  M.height = height;
  int size = M.width * M.height;
  M.elements = NULL;

  M.elements = (float*)malloc(size * sizeof(float));

  for (unsigned int i = 0; i < M.height * M.width; i++)
  {
	M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
  }
  return M;
}

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
  unsigned int data_read = MATRIX_SIZE*MATRIX_SIZE;
//  cutReadFilef(file_name, &(M->elements), &data_read, true);
  ifstream iFile(file_name);
  unsigned i = 0;
  if (iFile) {
	float data;
	while (iFile >> data) {
	  M->elements[i++] = data;
	}
  }
  data_read = i;
  return data_read;
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{
  //Interface host call to the device kernel code and invoke the kernel

  Matrix dM_M = AllocateDeviceMatrix(M);
  Matrix dM_N = AllocateDeviceMatrix(N);
  Matrix dM_P = AllocateDeviceMatrix(P);

  CopyToDeviceMatrix(dM_M, M);
  CopyToDeviceMatrix(dM_N, N);

  dim3 dimGrid, dimBlock;

  dimGrid.x = dimGrid.y = dimGrid.z = 1;
  dimBlock.x = dimBlock.y = MATRIX_SIZE;
  dimBlock.z = 1;

  MatrixMulKernel<<<dimGrid, dimBlock>>>(dM_M, dM_N, dM_P);

  CopyFromDeviceMatrix(P, dM_P);

  cudaFree(&dM_M);
  cudaFree(&dM_N);
  cudaFree(&dM_P);

}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
  Matrix Mdevice = M;
  int size = M.width * M.height * sizeof(float);
  cudaMalloc((void**)&Mdevice.elements, size);
  return Mdevice;
}

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
  int size = Mhost.width * Mhost.height * sizeof(float);
  Mdevice.height = Mhost.height;
  Mdevice.width = Mhost.width;
  Mdevice.pitch = Mhost.pitch;
  cudaMemcpy(Mdevice.elements, Mhost.elements, size,
	cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
  int size = Mdevice.width * Mdevice.height * sizeof(float);
  cudaMemcpy(Mhost.elements, Mdevice.elements, size,
	cudaMemcpyDeviceToHost);
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
void computeGold(float* C, const float* A, const float* B, unsigned int hA, 
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

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
//  cutWriteFilef(file_name, M.elements, M.width*M.height, 0.0001f);
  ofstream oFile(file_name);
  if (oFile) {
	for (int i = 0; i < MATRIX_SIZE; i++) {
	  oFile << M.elements[i] << " ";
	}
	oFile.close();
  }
}

