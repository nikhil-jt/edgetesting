# Edge Testing

### Setup

When you run the compiler.sh file, three seperate models will be generated, named after their respective platform. The compiler uses the bonnet compiler for the AIY Vision Kit as well as the edgetpu compiler for the Google Coral. The model used is mobilenet_v1_0.5_160 (both quantized and non-quantized), which can be found [here](https://www.tensorflow.org/lite/guide/hosted_models).

### Testing

After setting up each of the three platforms, download their respective python files to each machine. To test, run the python file using python3 with tensorflow and other required libraries. The model and label file paths are required as parameters, and for the vision kit, the output tensor key you have to specify is 'MobilenetV1/Predictions/Softmax' for the given models. An example for the vision kit is as follows:
'''sh
'''
