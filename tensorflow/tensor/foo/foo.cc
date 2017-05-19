#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"

using namespace tensorflow;

static Status ReadEntireFile(Env* env, const string& filename,
                             string* contents) {
  uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));
  contents->resize(file_size);
  std::unique_ptr<RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));
  StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(*contents)[0]));
  if (data.size() != file_size) {
    return errors::DataLoss("Truncated read of '", filename, "' expected ",
                            file_size, " got ", data.size());
  }
  if (data.data() != &(*contents)[0]) {
    memmove(&(*contents)[0], data.data(), data.size());
  }
  return Status::OK();
}

int main(int argc, char* argv[]) {
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Tensor* output = nullptr;
  // const DataType type = params_->op_kernel->output_type(index);
  // DCHECK(mutable_output(index) == nullptr);

  // Tensor* output = new Tensor();
  // AllocatorAttributes attr = output_alloc_attr(0);
  //  0, TensorShape({}),  &output)
  // Status s = allocate_tensor(0, TensorShape({}), &output, attr);

  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string file_name = "/tmp/grace_hopper.bmp";

  string o;
  ReadEntireFile(Env::Default(), "/tmp/grace_hopper.bmp", &o);
  std::cout << "size: " <<  o.size() << "\n";
  Tensor a(DT_STRING, TensorShape());
  a.scalar<string>()() = o;

  tensorflow::Output image_reader;
  
  Placeholder file_reader = Placeholder(root.WithOpName("input"), DataType::DT_STRING);
  
  if (tensorflow::StringPiece(file_name).ends_with(".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader);
  } else if (tensorflow::StringPiece(file_name).ends_with(".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader = Squeeze(root.WithOpName("squeeze_first_dim"),
                           DecodeGif(root.WithOpName("gif_reader"),
                                     file_reader));
  } else if (tensorflow::StringPiece(file_name).ends_with(".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader, DecodeJpeg::Channels(3));
  }

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "input", a},
  };

  // The session will initialize the outputs
  std::vector<tensorflow::Tensor> outputs;

  tensorflow::GraphDef graph;
  status = root.ToGraphDef(&graph);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  status = session->Create(graph);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  status = session->Run(inputs, {"bmp_reader"}, {}, &outputs);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Free any resources used by the session
  session->Close();
  return 0;
}
