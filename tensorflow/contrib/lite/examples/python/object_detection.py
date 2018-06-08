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
"""objection_detection for tflite"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import time
from heapq import heappush, nlargest

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

NUM_RESULTS = 1917
NUM_CLASSES = 91

X_SCALE = 10.0
Y_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

def load_box_priors(filename):
  with open(filename) as f:
    count = 0
    for line in f:
      row = line.strip().split(' ')
      box_priors.append(row)
      #print(box_priors[count][0])
      count = count + 1
      if count == 4:
        return

def load_labels(filename):
  my_labels = []
  input_file = open(filename, 'r')
  for l in input_file:
    my_labels.append(l.strip())
  return my_labels

def decode_center_size_boxes(locations):
  """calculate real sizes of boxes"""
  for i in range(0, NUM_RESULTS):
    ycenter = locations[i][0] / Y_SCALE * np.float(box_priors[2][i]) \
            + np.float(box_priors[0][i])
    xcenter = locations[i][1] / X_SCALE * np.float(box_priors[3][i]) \
            + np.float(box_priors[1][i])
    h = math.exp(locations[i][2] / H_SCALE) * np.float(box_priors[2][i])
    w = math.exp(locations[i][3] / W_SCALE) * np.float(box_priors[3][i])

    ymin = ycenter - h / 2.0
    xmin = xcenter - w / 2.0
    ymax = ycenter + h / 2.0
    xmax = xcenter + w / 2.0

    locations[i][0] = ymin
    locations[i][1] = xmin
    locations[i][2] = ymax
    locations[i][3] = xmax
  return locations

if __name__ == "__main__":
  file_name = "/tmp/grace_hopper.bmp"
  model_file = "/tmp/mobilenet_ssd.tflite"
  label_file = "/tmp/coco_labels_list.txt"
  box_prior_file = "/tmp/box_priors.txt"
  input_mean = 127.5
  input_std = 127.5
  floating_model = False
  show_image = False

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be classified")
  parser.add_argument("--graph", help=".tflite model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_mean", help="input_mean")
  parser.add_argument("--input_std", help="input standard deviation")
  parser.add_argument("--show_image", help="show_image")
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
  if args.show_image:
    show_image = args.show_image

  interpreter = interpreter_wrapper.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  #print(input_details)
  #print(output_details)

  # check the type of the input tensor
  if input_details[0]['dtype'] == type(np.float32(1.0)):
    floating_model = True

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(file_name)
  img = img.resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start_time = time.time()
  interpreter.invoke()
  finish_time = time.time()
  print("time spent:", ((finish_time - start_time) * 1000))

  box_priors = []
  load_box_priors(box_prior_file)
  labels = load_labels(label_file)
  predictions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
  output_classes = np.squeeze( \
                     interpreter.get_tensor(output_details[1]['index']))

  decode_center_size_boxes(predictions)

  heap = []
  for r in range(0, NUM_RESULTS):
    top_class_score = -1000.0
    top_class_score_index = -1

    for c in range(1, NUM_CLASSES):
      score = 1. / (1. + math.exp(-output_classes[r][c]))
      if score > top_class_score:
        top_class_score_inxdex = c
        top_class_score = score

    if top_class_score > 0.001:
      rect = (predictions[r][1] * width, predictions[r][0] * width, \
              predictions[r][3] * width, predictions[r][2] * width)
      heappush(heap, (output_classes[r][top_class_score_inxdex], r, \
                      labels[top_class_score_inxdex], rect))

  fig, ax = plt.subplots(1)

  ten = nlargest(10, heap)
  for e in ten:
    score = '{0:2.0f}%'.format(100. / (1. + math.exp(-e[0])))
    print(score, e[2], e[3])
    left, top, right, bottom = e[3]
    rect = patches.Rectangle((left, top), (right - left), (bottom - top), \
             linewidth=1, edgecolor='r', facecolor='none')

    if show_image:
      # Add the patch to the Axes
      ax.add_patch(rect)
      ax.text(left, top, e[2]+': '+score, fontsize=6,
              bbox=dict(facecolor='y', edgecolor='y', alpha=0.5))

  if show_image:
    ax.imshow(img)
    plt.show()
