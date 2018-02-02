#include "cuda_MP1.cuh"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Vector addition kernel thread specification
__global__ void VectorAddKernel(Vector A, Vector B, Vector C)
{
  //Add the two vectors
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<VSIZE) C.elements[i] = A.elements[i] + B.elements[i];

}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int cuda_MP1(int argc, char* argv[])
{

  // Vectors for the program
  Vector A;
  Vector B;
  Vector C;
  // Number of elements in the vectors
  unsigned int size_elements = VSIZE;
  int errorA = 0, errorB = 0;

  srand(2012);

  // Check command line for input vector files
  if (argc != 3 && argc != 4)
  {
	// No inputs provided
	// Allocate and initialize the vectors
	A = AllocateVector(VSIZE, 1);
	B = AllocateVector(VSIZE, 1);
	C = AllocateVector(VSIZE, 0);
  }
  else
  {
	// Inputs provided
	// Allocate and read source vectors from disk
	A = AllocateVector(VSIZE, 0);
	B = AllocateVector(VSIZE, 0);
	C = AllocateVector(VSIZE, 0);
	errorA = ReadFile(&A, argv[1]);
	errorB = ReadFile(&B, argv[2]);
	// check for read errors
	if (errorA != size_elements || errorB != size_elements)
	{
	  printf("Error reading input files %d, %d\n", errorA, errorB);
	  return 1;
	}
  }

  // A + B on the device
  VectorAddOnDevice(A, B, C);

  // compute the vector addition on the CPU for comparison
  Vector reference = AllocateVector(VSIZE, 0);
  computeGold(reference.elements, A.elements, B.elements, VSIZE);

  bool res = compareGold(reference.elements, C.elements, VSIZE, 0.0001f);
  printf("Test %s\n", (true == res) ? "PASSED" : "FAILED");

  // output result if output file is requested
  if (argc == 4)
  {
	WriteFile(C, argv[3]);
  }
  else if (argc == 2)
  {
	WriteFile(C, argv[1]);
  }

  // Free host matrices
  free(A.elements);
  A.elements = NULL;
  free(B.elements);
  B.elements = NULL;
  free(C.elements);
  C.elements = NULL;

  return 0;
}

// Allocate a vector of dimensions length
//	If init == 0, initialize to all zeroes.
//	If init == 1, perform random initialization.
Vector AllocateVector(int length, int init)
{
  Vector V;
  V.length = length;
  V.elements = NULL;

  V.elements = (float*)malloc(length * sizeof(float));

  for (unsigned int i = 0; i < V.length; i++)
  {
	V.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
  }
  return V;
}

// Read a floating point vector in from file
int ReadFile(Vector* V, char* file_name)
{
  unsigned int data_read = VSIZE;
//  cutReadFilef(file_name, &(V->elements), &data_read, true);
  ifstream iFile(file_name);
  unsigned i = 0;
  if (iFile) {
	float data;
	while (iFile >> data) {
	  V->elements[i++] = data;
	}
  }
  data_read = i;
  return data_read;
}

void VectorAddOnDevice(const Vector A, const Vector B, Vector C)
{
  Vector dV_A = AllocateDeviceVector(A);
  Vector dV_B = AllocateDeviceVector(B);
  Vector dV_C = AllocateDeviceVector(C);

  CopyToDeviceVector(dV_A, A);
  CopyToDeviceVector(dV_B, B);

  dim3 dimGrid, dimBlock;

  dimGrid.x = dimGrid.y = dimGrid.z = 1;
  dimBlock.x = BLOCK_SIZE;
  dimBlock.y = dimBlock.z = 1;

  VectorAddKernel<<<dimGrid, dimBlock>>>(dV_A, dV_B, dV_C);

  CopyFromDeviceVector(C, dV_C);

  cudaFree(&dV_A);
  cudaFree(&dV_B);
  cudaFree(&dV_C);

}

// Allocate a device vector of same size as V.
Vector AllocateDeviceVector(const Vector V)
{
  Vector Vdevice = V;
  int size = V.length * sizeof(float);
  cudaMalloc((void**)&Vdevice.elements, size);
  return Vdevice;
}

// Copy a host vector to a device vector.
void CopyToDeviceVector(Vector Vdevice, const Vector Vhost)
{
  int size = Vhost.length * sizeof(float);
  Vdevice.length = Vhost.length;
  cudaMemcpy(Vdevice.elements, Vhost.elements, size,
	cudaMemcpyHostToDevice);
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          vector A as provided to device
//! @param B          vector B as provided to device
//! @param N         length of vectors
////////////////////////////////////////////////////////////////////////////////
void computeGold(float* C, const float* A, const float* B, unsigned int N)
{
  for (unsigned int i = 0; i < N; ++i)
	C[i] = A[i] + B[i];
}

bool compareGold(float* ref, const float* C, unsigned int N, float precision)
{
  for (int i = 0; i < N; i++) {
	if (abs(ref[i] - C[i]) > precision) {
	  cout << i << ": " << ref[i] << ", " << C[i] << endl;
	  return false;
	}
  }
  return true;
}

// Copy a device vector to a host vector.
void CopyFromDeviceVector(Vector Vhost, const Vector Vdevice)
{
  int size = Vdevice.length * sizeof(float);
  cudaMemcpy(Vhost.elements, Vdevice.elements, size,
	cudaMemcpyDeviceToHost);
}

// Write a floating point vector to file
void WriteFile(Vector V, char* file_name)
{
//  cutWriteFilef(file_name, V.elements, V.length, 0.0001f);
  ofstream oFile(file_name);
  if (oFile) {
	for (int i = 0; i < VSIZE; i++) {
	  oFile << V.elements[i] << " ";
	}
	oFile.close();
  }
}
