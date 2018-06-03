# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from PIL import Image

from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

def load_labels(filename):
  labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    labels.append(l.strip())
  return labels

if __name__ == "__main__":
  file_name = "/tmp/grace_hopper.bmp"
  model_file = "/tmp/mobilenet_v1_1.0_224_quant.tflite"
  label_file = "/tmp/labels.txt"
  input_mean = 127.5
  input_std = 127.5
 
  floating_model = False

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be classified")
  parser.add_argument("--graph", help=".tflite model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_mean", help="input_mean")
  parser.add_argument("--input_std", help="input standard deviation")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std

  interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  if (input_details[0]['dtype'] == type(np.float32(1.0))):
    floating_model = True
    print(input_details[0]['dtype'] == type(np.float32(1.0)))

  #print(input_details)
  #print(output_details)

  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  #print(height, width)
  img = Image.open(file_name)
  img = img.resize((width, height))

  input_data = np.expand_dims(img, axis=0)

  if (floating_model):
   input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])

  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(label_file)
  for i in top_k:
    if (floating_model):
      print(labels[i]+":", results[i])
    else:
      print(labels[i]+":", results[i]/255.0)

