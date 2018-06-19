0. get pre-trained MobileNet SSD model(s), e.g.,
```
curl http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz | tar xzv -C /tmp

bazel run --config=opt   //tensorflow/contrib/lite/toco:toco -- --input_file=/tmp/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb  --output_file=/tmp/ssd_mobilenet_v1_coco.tflite --inference_type=FLOAT --input_shape=1,300,300,3 --input_array=Preprocessor/sub --output_arrays=concat,concat_1
```
or 
```
curl http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz | tar xzv -C /tmp

bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- --input_file=/tmp/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --output_file=/tmp/ssd_mobilenet_v2_coco.tflite --inference_type=FLOAT --input_shape=1,300,300,3 --input_array=Preprocessor/sub --output_arrays=concat,concat_1
```

1. prepare files
  a. `cp ${TF_ROOT}/tensorflow/contrib/lite/examples/android/assets/box_priors.txt /tmp/`
  b. `cp ${TF_ROOT}/tensorflow/contrib/lite/examples/android/assets/coco_labels_list.txt /tmp/`
  c. `cp ${TF_MODELS}/research/object_detection/test_images/image2.jpg /tmp/`

2. run it
```
python3 tensorflow/contrib/lite/examples/python/object_detection.py --graph /tmp/ssd_mobilenet_v1_coco.tflite   --image /tmp/image2.jpg  --show_image True
```
