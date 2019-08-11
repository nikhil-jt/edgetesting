import tensorflow as tf
import cv2
import argparse
import numpy as np
from datetime import datetime

def load_labels(filename):
    def split(line):
        return tuple(word.strip() for word in line.split(','))

    with open(filename, encoding='utf-8') as f:
        return tuple(split(line) for line in f)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required = True)
parser.add_argument('--input', required=True)
parser.add_argument('--input_size', type=int, required=True)
parser.add_argument('--label_path', required=True)
args = parser.parse_args()

interpreter = tf.lite.Interpreter(model_path = args.model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (args.input_size, args.input_size))

interpreter.set_tensor(input_details[0]['index'], [image])

starttime=datetime.now()
interpreter.invoke()
endtime = datetime.now()
deltatime = endtime-starttime
print(str(deltatime.seconds) + "s, " + str(deltatime.microseconds/1000) + "ms") 

results = interpreter.get_tensor(output_details[0]['index'])
pairs = [pair for pair in enumerate(results[0]) if pair[1] > 0.1]
pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
pairs = pairs[0:5]
_CLASSES = load_labels(args.label_path)
classes =  [('/'.join(_CLASSES[index]), prob) for index, prob in pairs]

for i, (label, score) in enumerate(classes):
    print('Result %d: %s (prob=%f)' % (i, label, score))
