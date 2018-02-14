/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <sys/time.h>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/string_util.h"

#define LOG(x) std::cerr

namespace tflite {
namespace benchmark_ops {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static double benchmark_conv_tflite(
    int N,
    int C,
    int H,
    int W,
    int K,
    int kernel,
    int warmup = 5,
    int run = 10,
    bool floating = false,
    bool nnapi = false)
{
  std::unique_ptr<Interpreter> interpreter(new Interpreter);

  int base_index = 0;

  // two inputs: input, filter, and bias
  interpreter->AddTensors(3, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);

  // set input and output tensors
  interpreter->SetInputs({0, 1, 2});
  interpreter->SetOutputs({3});

  // set parameters of tensors
  TfLiteQuantizationParams quant;

  if (floating) {
    interpreter->SetTensorParametersReadWrite(
      0, kTfLiteFloat32, "input",
      {N, H, W, C}, quant);
    interpreter->SetTensorParametersReadWrite(
      1, kTfLiteFloat32, "filter", 
      {K, kernel, kernel, C}, quant);
    interpreter->SetTensorParametersReadWrite(
      2, kTfLiteFloat32, "bias", 
      {K}, quant);
    interpreter->SetTensorParametersReadWrite(
      3, kTfLiteFloat32, "output",
      {N, H, W, C}, quant);
  } else {
    quant.scale = 1.0;
    interpreter->SetTensorParametersReadWrite(
      0, kTfLiteUInt8, "input",
      {N, H, W, C}, quant);
    interpreter->SetTensorParametersReadWrite(
      1, kTfLiteUInt8, "filter", 
      {K, kernel, kernel, C}, quant);
    quant.scale = 1.0;
    interpreter->SetTensorParametersReadWrite(
      2, kTfLiteInt32, "bias", 
      {K}, quant);
    quant.scale = 100.0;
    interpreter->SetTensorParametersReadWrite(
      3, kTfLiteUInt8, "output",
      {N, H, W, C}, quant);
  }

  ops::builtin::BuiltinOpResolver resolver;
	  TfLiteRegistration* conv2d_op =
      resolver.FindOp(BuiltinOperator_CONV_2D);
  auto* params = reinterpret_cast<TfLiteConvParams*>(
      malloc(sizeof(TfLiteConvParams)));

  params->padding = kTfLitePaddingSame;
  params->stride_width = 1;
  params->stride_height = 1;
  params->activation = kTfLiteActNone;
  
  interpreter->AddNodeWithParameters({0, 1, 2}, {3}, nullptr, 0, params,
                                     conv2d_op, nullptr);

  interpreter->SetNumThreads(4);
  interpreter->UseNNAPI(nnapi);
  interpreter->AllocateTensors();

  struct timeval start, stop;

  for (int i=0; i < warmup; i++)
    interpreter->Invoke();

  gettimeofday(&start, NULL); 
  for (int i=0; i < run; i++)
    interpreter->Invoke();
  gettimeofday(&stop, NULL); 
  return ((get_us(stop) - get_us(start)) / (run * 1000));
}

void run_convolutions()
{
  int warmup = 2, mainrun = 10;
  // float32
  for (int space : {14, 26, 52, 104}) {
    for (int input_channel : {64, 128, 256, 512}) {
      for (int kernel : {1, 3}) {
        int output_channel = input_channel;
        const double cpu_time_int = benchmark_conv_tflite(
            1,
            input_channel,
            space,
            space,
            output_channel,
            kernel,
            warmup,
            mainrun);

        const double cpu_time_float = benchmark_conv_tflite(
            1,
            input_channel,
            space,
            space,
            output_channel,
            kernel,
            warmup,
            mainrun, true);

        const double cpu_time_int_nnapi = benchmark_conv_tflite(
            1,
            input_channel,
            space,
            space,
            output_channel,
            kernel,
            warmup,
            mainrun, false, true);

        const double cpu_time_float_nnapi = benchmark_conv_tflite(
            1,
            input_channel,
            space,
            space,
            output_channel,
            kernel,
            warmup,
            mainrun, true, true);

       const double flops = double(input_channel) * output_channel * kernel *
            kernel * (kernel == 1 ? space : space - 2) *
            (kernel == 1 ? space : space - 2) * 2;

       printf(
            "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t"
            "8b-lite GOPS: %.2f\t"
            "32b-lite GFLOPS: %.2f\t"
            "8b-nnapi GOPS: %.2f\t"
            "32b-nnapi GFLOPS: %.2f\n",
            space,
            space,
            input_channel,
            output_channel,
            kernel,
            kernel,
            flops / cpu_time_int / 1E6,
            flops / cpu_time_float / 1E6,
            flops / cpu_time_int_nnapi / 1E6,
            flops / cpu_time_float_nnapi / 1E6);
      }
    }
  }
}

void run_depthwise_separable_convolutions()
{
}

int Main(int argc, char** argv) {
  run_convolutions();
  run_depthwise_separable_convolutions();
  return 0;
}

}  // namespace benchmark_ops
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::benchmark_ops::Main(argc, argv);
}
