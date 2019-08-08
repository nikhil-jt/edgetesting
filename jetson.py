import tensorflow as tf
import cv2
import argparse
import numpy as np
from datetime import datetime
import time
def load_labels(filename):
    def split(line):
        return tuple(word.strip() for word in line.split(','))

    with open(filename, encoding='utf-8') as f:
        return tuple(split(line) for line in f)

def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.io.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

# The TensorRT inference graph file downloaded from Colab or your local machine.
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required = True)
parser.add_argument('--tensor_debug', default = False)
parser.add_argument('--input', required=True)
parser.add_argument('--input_key', required=True)
parser.add_argument('--output_key', required=True)
parser.add_argument('--input_size', type=int, required=True)
parser.add_argument('--label_path', required=True)
args = parser.parse_args()
#pb_fname = "tf_models/mobilenet_v1_160res_0.5_imagenet.pb"
trt_graph = get_frozen_graph(args.model_path)

#input_names = ['image_tensor']

# Create session and load graph
tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.compat.v1.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')
if args.tensor_debug:
    for op in tf_sess.graph.get_operations():
        print(str(op.name))
tf_input = tf_sess.graph.get_tensor_by_name(args.input_key + ':0')
tf_results = tf_sess.graph.get_tensor_by_name(args.output_key + ':0')
#image = cv2.imread(args.input)/256
image = cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB)/255
image = cv2.resize(image, (args.input_size, args.input_size))
starttime = datetime.now()
results = tf_sess.run([tf_results], feed_dict={tf_input:[image]})
endtime = datetime.now()
deltatime = endtime-starttime
print(str(deltatime.seconds) + "s, " + str(deltatime.microseconds/1000) + "ms")
#print(results[0])
pairs = [pair for pair in enumerate(tuple(results[0][0])) if pair[1] > 0.1]
#print(pairs)
pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
pairs = pairs[0:5]
#print(pairs)
_CLASSES = load_labels(args.label_path)
#print(len(_CLASSES))
classes =  [('/'.join(_CLASSES[index]), prob) for index, prob in pairs]
for i, (label, score) in enumerate(classes):
    print('Result %d: %s (prob=%f)' % (i, label, score))

