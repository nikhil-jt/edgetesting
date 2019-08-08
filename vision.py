#!/usr/bin/env python3
#
# Copyright 2017 Google Inc.
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
"""Script to load and run model on Vision Bonnet.

The primary purpose of this script is to make sure a compiled model can run on
Vision Bonnet. It does not try to interpret the output tensor.

Example:
~/AIY-projects-python/src/examples/vision/any_model_camera.py \
  --model_path ~/models/mobilenet_ssd_256res_0.125_person_cat_dog.binaryproto \
  --input_height 256 \
  --input_width 256
"""
import argparse
import numpy as np
from picamera import PiCamera
from aiy.vision.models import image_classification
from PIL import Image
from aiy.vision.inference import ImageInference, ModelDescriptor
from aiy.vision.models import utils
from datetime import datetime
def tensors_info(tensors):
    return ', '.join('%s [%d elements]' % (name, len(tensor.data))
        for name, tensor in tensors.items())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='test_model', help='Model identifier.')
    parser.add_argument('--label_path', required = True, help='Path to label file.')
    parser.add_argument('--model_path', required=True, help='Path to model file.')
    parser.add_argument('--input', required=True, help='Input height.')
    parser.add_argument('--input_size', type=int, required=True, help='Input height.')
    parser.add_argument('--tensor_key', default = 'MobilenetV1/Predictions/Softmax')
    args = parser.parse_args()
    image = Image.open(args.input)
    width, height = image.size
    model = ModelDescriptor(
        name=args.model_name,
        input_shape=(1, args.input_size, args.input_size, 3),
        input_normalizer=(128, 128),
        compute_graph=utils.load_compute_graph(args.model_path))
    inference = ImageInference(model)
    if inference:
        starttime = datetime.now()
        result = inference.run(image)
        for key in result.tensors:
            print(key)
        #print(result.tensors.keys())
#        print(result.tensors["prediction"])
        deltatime = datetime.now()-starttime
        print(str(deltatime.seconds) + "s " + str(deltatime.microseconds/1000) + "ms")
        assert len(result.tensors) == 1
        tensor = result.tensors[args.tensor_key]
        #print(tensor)
       # print(tensor.shape)
#    assert utils.shape_tuple(tensor.shape) == (1, 1, 1, len(_CLASSES))
        probs = tuple(tensor.data)
        pairs = [pair for pair in enumerate(probs) if pair[1] > 0.1]
        print(pairs)
        pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
        pairs = pairs[0:5]
        print(pairs)
        _CLASSES = utils.load_labels(args.label_path)
        print(len(_CLASSES))
        classes =  [('/'.join(_CLASSES[index]), prob) for index, prob in pairs]

        for i, (label, score) in enumerate(classes):
            print('Result %d: %s (prob=%f)' % (i, label, score))



if __name__ == '__main__':
    main()
