#include "MP_test_functions.h"

int MP0_function_wrapper()
{
  cout << "Starting MP0 test!" << endl;
  cuda_MP0();
  cout << "MP0 test ended!\n" << endl;
  return 0;
}

int MP1_function_wrapper(int argc, char* argv[])
{
  cout << "Starting MP1 test!" << endl;
  cuda_MP1(argc, argv);
  cout << "MP1 test ended!\n" << endl;
  return 0;
}

int MP2_function_wrapper(int argc, char* argv[])
{
  cout << "Starting MP2 test!" << endl;
  cuda_MP2(argc, argv);
  cout << "MP2 test ended!\n" << endl;
  return 0;
}

int MP3_function_wrapper(int argc, char* argv[])
{
  cout << "Starting MP3 test!" << endl;
  cuda_MP3(argc, argv);
  cout << "MP3 test ended!\n" << endl;
  return 0;
}

int MP4_function_wrapper(int argc, char* argv[])
{
  cout << "Starting MP4 test!" << endl;
  cuda_MP4(argc, argv);
  cout << "MP4 test ended!\n" << endl;
  return 0;
}

int MP5_function_wrapper(int argc, char* argv[])
{
  cout << "Starting MP5 test!" << endl;
  cuda_MP5(argc, argv);
  cout << "MP5 test ended!\n" << endl;
  return 0;
}

int MP6_function_wrapper(int argc, char* argv[])
{
  cout << "Starting MP6 test!" << endl;
  cuda_MP6(argc, argv);
  cout << "MP6 test ended!\n" << endl;
  return 0;
}

int MP7_function_wrapper(int argc, char* argv[])
{
  cout << "Starting MP7 test!" << endl;
//  cuda_MP7(argc, argv);

  histo_cuda();
  cout << "MP7 test ended!\n" << endl;
  return 0;
}