#ifndef CUDA_MP1_VECTORADD_H
#define CUDA_MP1_VECTORADD_H

// Thread block size
#define BLOCK_SIZE 256

// Vector dimensions
#define VSIZE 256 // vector size

// Vector Structure declaration
typedef struct {
  //length of the vector
  unsigned int length;
  //Pointer to the first element of the vector
  float* elements;
} Vector;


#endif // !CUDA_MP1_VECTORADD_H
