#include "test_functions.h"

int cuda_query_function0()
{
  cout << "Starting CUDA device query: " << endl;
  
  int dev_count; 
  cudaGetDeviceCount(&dev_count);
  cout << "Total number of CUDA devices: " << dev_count << "\n" << endl;

  cudaDeviceProp dev_prop;
  for (int i = 0; i < dev_count; i++) {
	cout << "CUDA device " << i << endl;
	cudaGetDeviceProperties(&dev_prop, i);
	cout << "Max # of threads/block: " << dev_prop.maxThreadsPerBlock << endl;
	cout << "# of SMs: " << dev_prop.multiProcessorCount << endl;
	cout << "Device clock frequency: " << dev_prop.clockRate << "Hz" << endl;
	cout << "Max allowed threads in each dimension (x, y, z): (" <<
	  dev_prop.maxThreadsDim[0] << ", " << dev_prop.maxThreadsDim[1] << ", "
	  << dev_prop.maxThreadsDim[2] << ")" << endl;
	cout << "Max allowed grid size in each dimension (x, y, z): (" <<
	  dev_prop.maxGridSize[0] << ", " << dev_prop.maxGridSize[1] << ", "
	  << dev_prop.maxGridSize[2] << ")" << endl;
	cout << "Device warp size: " << dev_prop.warpSize << endl;
	cout << "Registers per block: " << dev_prop.regsPerBlock << "B " << endl;
	cout << "Shared memory per block: " << dev_prop.sharedMemPerBlock << "B"
	  << endl;
	cout << "Total constant memory: " << dev_prop.totalConstMem << "B" << endl;
  }

  cout << "\nCUDA device query end. \n" << endl;

  cout << "Starting CUDA random matrix multiplication "
    "(and compared with serial version)." << endl;

  srand(time(NULL));
  /*
  auto a = rand() % 100;
  auto b = rand() % 100;
  auto c = rand() % 100;
  printf("Initialized numbers: %d, %d, %d\n", a, b, c);
  */

  unsigned int n = 20;
  unsigned int size = n*n;

  auto A = new int[size];
  auto B = new int[size];
  auto C_h = new int[size];
  auto C_d = new int[size];

  for (size_t i = 0; i < size; i++) {
	A[i] = rand() % 100;
	B[i] = rand() % 100;
  }

  // serial matrix calculation
  SerialMultiply(A, B, C_h, n);
  ParallelMultiply(A, B, C_d, n);

  //  MatrixDisplay(A, n, n);
  //  MatrixDisplay(B, n, n);
  //  MatrixDisplay(C_h, n, n);
  // cout << "By device: " << endl;
  //  MatrixDisplay(C_d, n, n);

  delete[] A;
  delete[] B;
  delete[] C_h;
  delete[] C_d;

  cout << "Class MyMatrix testing: " << endl;

  MyMatrix MA(3, 5);
  MA.MatrixDisplay();

  cout << "CUDA random matrix multiplication completed!" << endl;

  return 0;
}