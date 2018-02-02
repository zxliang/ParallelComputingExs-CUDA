#include "matrix_functionality.h"

int SerialMultiply(int *mtrx_a, int *mtrx_b, int *mtrx_c, int n)
{
  if (mtrx_a == NULL || mtrx_b == NULL) {
	cout << "One of the input matrices pointer is NULL" << endl;
	return -1;
  }

  int sum = 0;
  for (int i = 0; i < n; i++) {
	for (int j = 0; j < n; j++) {
	  sum = 0;
	  for (int k = 0; k < n; k++) {
		sum += mtrx_a[i * n + k] * mtrx_b[k * n + j];
	  }
	  mtrx_c[i * n + j] = sum;
	}
  }
  return 0;
}
