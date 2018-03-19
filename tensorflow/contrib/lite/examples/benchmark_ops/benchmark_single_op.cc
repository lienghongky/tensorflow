/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include <getopt.h>
#include <sys/time.h>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/kernels/register.h"

#define LOG(x) std::cerr

namespace tflite {
namespace benchmark_ops {

struct Settings {
  bool flops = false;
  bool profiling = false;
  int mainrun = 10;
  int warmup = 2;
  int number_of_threads = 4;
};

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static double benchmark_conv_tflite(int N, int C, int H, int W, int K,
                                    int kernel, Settings* s,
                                    bool floating = false, bool nnapi = false,
                                    bool dwise = false) {
  std::unique_ptr<Interpreter> interpreter(new Interpreter);

  int base_index = 0;

  // two inputs: input, filter, and bias
  interpreter->AddTensors(3, &base_index);
  // one output
  interpreter->AddTensors(1, &base_index);

  // set input and output tensors
  if (nnapi && floating)
    interpreter->SetInputs({0});
  else
    interpreter->SetInputs({0, 1, 2});

  interpreter->SetOutputs({3});

  // set parameters of tensors
  TfLiteQuantizationParams quant;

  if (floating) {
    interpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input",
                                              {N, H, W, C}, quant);
    if (!nnapi)
      interpreter->SetTensorParametersReadWrite(1, kTfLiteFloat32, "filter",
                                                {K, kernel, kernel, C}, quant);
    else {
      float* f = new float[K * kernel * kernel * C]();
      interpreter->SetTensorParametersReadOnly(
          1, kTfLiteFloat32, "filter", {K, kernel, kernel, C}, quant, (char*)f,
          K * kernel * kernel * C * 4);
    }

    if (!nnapi)
      interpreter->SetTensorParametersReadWrite(2, kTfLiteFloat32, "bias", {K},
                                                quant);
    else {
      float* g = new float[K]();
      interpreter->SetTensorParametersReadOnly(2, kTfLiteFloat32, "bias", {K},
                                               quant, (char*)g, K * 4);
    }
    interpreter->SetTensorParametersReadWrite(3, kTfLiteFloat32, "output",
                                              {N, H, W, C}, quant);
  } else {
    quant.scale = 1.0;
    interpreter->SetTensorParametersReadWrite(0, kTfLiteUInt8, "input",
                                              {N, H, W, C}, quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteUInt8, "filter",
                                              {K, kernel, kernel, C}, quant);
    quant.scale = 1.0;
    interpreter->SetTensorParametersReadWrite(2, kTfLiteInt32, "bias", {K},
                                              quant);
    quant.scale = 100.0;
    interpreter->SetTensorParametersReadWrite(3, kTfLiteUInt8, "output",
                                              {N, H, W, C}, quant);
  }

  ops::builtin::BuiltinOpResolver resolver;
  TfLiteRegistration* conv2d_op;
  if (!dwise) {
    conv2d_op = resolver.FindOp(BuiltinOperator_CONV_2D);
    auto* params =
        reinterpret_cast<TfLiteConvParams*>(malloc(sizeof(TfLiteConvParams)));

    params->padding = kTfLitePaddingSame;
    params->stride_width = 1;
    params->stride_height = 1;
    params->activation = kTfLiteActNone;

    interpreter->AddNodeWithParameters({0, 1, 2}, {3}, nullptr, 0, params,
                                       conv2d_op, nullptr);

  } else {
    conv2d_op = resolver.FindOp(BuiltinOperator_DEPTHWISE_CONV_2D);
    auto* params = reinterpret_cast<TfLiteDepthwiseConvParams*>(
        malloc(sizeof(TfLiteDepthwiseConvParams)));

    params->padding = kTfLitePaddingSame;
    params->stride_width = 1;
    params->stride_height = 1;
    params->depth_multiplier = 1.0;
    params->activation = kTfLiteActNone;
    interpreter->AddNodeWithParameters({0, 1, 2}, {3}, nullptr, 0, params,
                                       conv2d_op, nullptr);
  }

  interpreter->SetProfiling(s->profiling);
  interpreter->SetNumThreads(s->number_of_threads);
  interpreter->UseNNAPI(nnapi);
  interpreter->AllocateTensors();

  struct timeval start, stop;

  for (int i = 0; i < s->warmup; i++) interpreter->Invoke();

  gettimeofday(&start, NULL);
  for (int i = 0; i < s->mainrun; i++) interpreter->Invoke();
  gettimeofday(&stop, NULL);
  return ((get_us(stop) - get_us(start)) / (s->mainrun * 1000));
}

void run_convolutions(Settings* s) {
  // for (int space : {14, 26, 52, 104}) {
  for (int space : {104}) {
    // for (int input_channel : {64, 128, 256, 512}) {
    for (int input_channel : {64}) {
      // for (int kernel : {1, 3}) {
      for (int kernel : {3}) {
        int output_channel = input_channel;
        const double cpu_time_int = benchmark_conv_tflite(
            1, input_channel, space, space, output_channel, kernel, s);

        const double cpu_time_float = benchmark_conv_tflite(
            1, input_channel, space, space, output_channel, kernel, s, true);

        const double cpu_time_int_nnapi =
            benchmark_conv_tflite(1, input_channel, space, space,
                                  output_channel, kernel, s, false, true);

        const double cpu_time_float_nnapi =
            benchmark_conv_tflite(1, input_channel, space, space,
                                  output_channel, kernel, s, true, true);

        const double flops = double(input_channel) * output_channel * kernel *
                             kernel * (kernel == 1 ? space : space - 2) *
                             (kernel == 1 ? space : space - 2) * 2;
        if (s->flops) {
          printf(
              "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t"
              "8bTfLite GOPS: %.2f\t"
              "32bTfLite GFLOPS: %.2f\t"
              "8bNNAPI GOPS: %.2f\t"
              "32bNNAPI GFLOPS: %.2f\n",
              space, space, input_channel, output_channel, kernel, kernel,
              flops / cpu_time_int / 1E6, flops / cpu_time_float / 1E6,
              flops / cpu_time_int_nnapi / 1E6,
              flops / cpu_time_float_nnapi / 1E6);
        } else
          printf(
              "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t"
              "8bTfLite time: %.2f\t"
              "32bTfLite time: %.2f\t"
              "8bNNAPI time: %.2f\t"
              "32bNNAPI time: %.2f\n",
              space, space, input_channel, output_channel, kernel, kernel,
              cpu_time_int, cpu_time_float, cpu_time_int_nnapi,
              cpu_time_float_nnapi);
      }
    }
  }
}

void run_depthwise_separable_convolutions(Settings* s) {
  for (int space : {14, 26, 52, 104}) {
    for (int channel : {64, 128, 256, 512}) {
      for (int kernel : {3}) {
        const double cpu_time_int = benchmark_conv_tflite(
            1, channel, space, space, channel, kernel, s, false, false, true);

        const double cpu_time_float = benchmark_conv_tflite(
            1, channel, space, space, channel, kernel, s, true, false, true);

        const double cpu_time_int_nnapi = benchmark_conv_tflite(
            1, channel, space, space, channel, kernel, s, false, true, true);

        const double cpu_time_float_nnapi = benchmark_conv_tflite(
            1, channel, space, space, channel, kernel, s, true, true, true);

        const double dwise_bandwidth =
            sizeof(float) * double(channel) *
            (2 * (space - 2) * (space - 2) + kernel * kernel);

        if (s->flops) {
          printf(
              "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t"
              "8bTfLite Dwise GiB/s: %.2f\t"
              "32bTfLite Dwise GiB/s: %.2f\t"
              "8bNNAPI Dwise GiB/s: %.2f\t"
              "32bNNAPI Dwise GiB/s: %.2f\n",
              space, space, channel, channel, kernel, kernel,
              dwise_bandwidth / sizeof(float) / cpu_time_int / 1E6,
              dwise_bandwidth / cpu_time_float / 1E6,
              dwise_bandwidth / sizeof(float) / cpu_time_int_nnapi / 1E6,
              dwise_bandwidth / cpu_time_float_nnapi / 1E6);
        } else {
          printf(
              "Conv: X: %ix%i  \tC: %i -> %i\tK: %ix%i\t"
              "8bTfLite Dwise time: %.2f\t"
              "32bTfLite Dwise time: %.2f\t"
              "8bNNAPI Dwise time: %.2f\t"
              "32bNNAPI Dwise time: %.2f\n",
              space, space, channel, channel, kernel, kernel, cpu_time_int,
              cpu_time_float, cpu_time_int_nnapi, cpu_time_float_nnapi);
        }
      }
    }
  }
}

void display_usage() {
  LOG(INFO) << "benchmark_ops\n"
            << "--flops, -f: flops\n"
            << "--profiling, -p: profiling\n"
            << "--mainrun, -r: number of main runs\n"
            << "--warmup, -w: number of warmup runs\n"
            << "--threads, -t: number of threads\n"
            << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"flops", required_argument, 0, 'f'},
        {"profiling", required_argument, 0, 'p'},
        {"mainrun", required_argument, 0, 'r'},
        {"threads", required_argument, 0, 't'},
        {"warmup", required_argument, 0, 'w'},
        {0, 0, 0, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "f:p:r:t:w:", long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'f':
        s.flops = atoi(optarg);
        break;
      case 'p':
        s.profiling = atoi(optarg);
        break;
      case 'r':
        s.mainrun = atoi(optarg);
        break;
      case 't':
        s.number_of_threads = atoi(optarg);
        break;
      case 'w':
        s.warmup = atoi(optarg);
        break;
      case 'h':
      case '?':
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  run_convolutions(&s);
  // run_depthwise_separable_convolutions(&s);
  return 0;
}

}  // namespace benchmark_ops
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::benchmark_ops::Main(argc, argv);
}
