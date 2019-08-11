import argparse
import numpy as np
from picamera import PiCamera
from aiy.vision.models import image_classification
from PIL import Image
from aiy.vision.inference import ImageInference, ModelDescriptor
from aiy.vision.models import utils
from datetime import datetime

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
    deltatime = datetime.now()-starttime
    print(str(deltatime.seconds) + "s " + str(deltatime.microseconds/1000) + "ms")
    
    assert len(result.tensors) == 1
    tensor = result.tensors[args.tensor_key]
    
    probs = tuple(tensor.data)
    pairs = [pair for pair in enumerate(probs) if pair[1] > 0.1]
    pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    pairs = pairs[0:5]
    _CLASSES = utils.load_labels(args.label_path)
    classes =  [('/'.join(_CLASSES[index]), prob) for index, prob in pairs]
    
    for i, (label, score) in enumerate(classes):
        print('Result %d: %s (prob=%f)' % (i, label, score))
