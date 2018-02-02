#include "test_functions.h"
#include "MP_test_functions.h"

int main(int argc, char* argv[])
{
  cuda_query_function0();

  cout << "\nThe following are MP functions:\n " << endl;

//  MP0_function_wrapper();
//  MP1_function_wrapper(argc, argv);
//  MP2_function_wrapper(argc, argv);
//  MP3_function_wrapper(argc, argv);
//  MP4_function_wrapper(argc, argv);
//  MP5_function_wrapper(argc, argv);
//  MP6_function_wrapper(argc, argv);
  MP7_function_wrapper(argc, argv);

  return 0;
}


