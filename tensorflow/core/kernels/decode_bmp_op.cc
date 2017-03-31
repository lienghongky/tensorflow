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

// See docs in ../ops/image_ops.cc

#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Decode the contents of a BMP file
class DecodeBmpOp : public OpKernel {
 public:
  explicit DecodeBmpOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("channels", &channels));
    OP_REQUIRES(
        context, channels == 0 || channels == 3,
        errors::InvalidArgument("channels must be 0 or 3, got ", channels));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& contents = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(contents.shape()),
                errors::InvalidArgument("contents must be scalar, got shape ",
                                        contents.shape().DebugString()));

    // Start decoding image to get shape details
    const StringPiece input = contents.scalar<string>()();

    bool top_down = false;
    uint8* const img_bytes = (uint8*)input.data();
    const int header_size = *(reinterpret_cast<int*>(img_bytes + 10));
    const int width = *(reinterpret_cast<int*>(img_bytes + 18));
    const int height = *(reinterpret_cast<int*>(img_bytes + 22));

    // if height is negative, data layout is top down
    // otherwise, it's bottom up
    top_down = (height < 0) ? true : false;

    // Decode image, allocating tensor once the image size is known
    Tensor* output = nullptr;
    const ::tensorflow::Status _status = context->allocate_output(
        0, TensorShape({abs(height), width, 3}), &output);
    if (TF_PREDICT_FALSE(!_status.ok())) return;

    uint8* const bmp_pixels = &img_bytes[header_size];

    Decode(bmp_pixels, output->flat<uint8>().data(), width, abs(height),
           top_down);
  }

  uint8* Decode(uint8* const input, uint8* const output, const int width,
                const int height, bool top_down);

 private:
  int channels;
};
REGISTER_KERNEL_BUILDER(Name("DecodeBmp").Device(DEVICE_CPU), DecodeBmpOp);

uint8* DecodeBmpOp::Decode(uint8* const input, uint8* const output,
                           const int width, const int height, bool top_down) {
  // there may be padding bytes when the width is not a multiple of 4 bytes
  int row_size = (8 * width * 3 + 31) / 32 * 4;

  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down)
        src_pos = ((height - 1 - i) * row_size) + j * 3;
      else
        src_pos = i * row_size + j * 3;

      dst_pos = (i * width + j) * 3;
      output[dst_pos] = input[src_pos + 2];
      output[dst_pos + 1] = input[src_pos + 1];
      output[dst_pos + 2] = input[src_pos];
    }
  }

  return output;
}

}  // namespace tensorflow
