#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "tensorflow/compiler/aot/tests/test_graph_mobilenetv1.h"

using namespace std;

void get_file_to_float(string fn, float bmp_float[][224][3])
{
  ifstream file(fn, ios::in|ios::binary|ios::ate);
  uint8_t bmp[224][224][3];

  if (file.is_open())
    file.read((char *)bmp, 224*224*3);

  for (int i; i < 224; i++)
    for (int j; j < 224; j++)
      for (int k; k < 3; k++)
        bmp_float[i][j][j] = (bmp[i][j][k] - 127.5f) / 127.5f;
}

int main(int argc, char *argv[])
{
  string foo = "/tmp/grace_hopper.rgb";
  float bmp_float[224][224][3];

  get_file_to_float(foo, bmp_float);

  Eigen::ThreadPool tp(8);  // Size the thread pool as appropriate.
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());

  foobar::MobilenetV1 mobilenetv1;
  mobilenetv1.set_thread_pool(&device);

  mobilenetv1.set_arg0_data(bmp_float);

  int64_t start, stop;
  start  = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  for (int i=0; i < 100; i++)
    mobilenetv1.Run();
  stop  = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

  cout << "time diff: " << (stop - start)/100 << " (ms)\n";
}
