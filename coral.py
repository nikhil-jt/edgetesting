from edgetpu.classification.engine import ClassificationEngine
import argparse
from datetime import datetime
import PIL
def load_labels(filename):
    def split(line):
        return tuple(word.strip() for word in line.split(','))

    with open(filename, encoding='utf-8') as f:
        return tuple(split(line) for line in f)
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required = True)
parser.add_argument("--input", required = True)
parser.add_argument("--label_path", required = True)
args = parser.parse_args()
engine = ClassificationEngine(args.model_path)
image = PIL.Image.open(args.input)
starttime = datetime.now()
result = engine.ClassifyWithImage(image, top_k=5)
endtime = datetime.now()
deltatime = endtime-starttime
print(str(deltatime.seconds) + "s, " + str(deltatime.microseconds/1000) + "ms")
_CLASSES = load_labels(args.label_path)
classes =  [('/'.join(_CLASSES[index]), prob) for index, prob in result]
for i, (label, score) in enumerate(classes):
    print('Result %d: %s (prob=%f)' % (i, label, score))

