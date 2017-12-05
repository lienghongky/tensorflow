package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/contrib/lite:build_def.bzl", "tflite_linkopts")

tf_cc_binary(
    name = "label_image",
    srcs = ["label_image.cc"],
    deps = [
        ":bitmap_helpers",
        "//tensorflow/contrib/lite/kernels:builtin_ops",
    ],
    linkopts = tflite_linkopts() + ["-lm"],
)

cc_library(
    name = "bitmap_helpers",
    srcs = ["bitmap_helpers.cc"],
    hdrs = ["bitmap_helpers.h"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
