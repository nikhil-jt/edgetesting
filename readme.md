# Edge Testing

### Setup

When you run the compiler.sh file, three seperate models will be generated, named after their respective platform. The compiler uses the bonnet compiler for the AIY Vision Kit as well as the edgetpu_compiler for the Google Coral. The model used is mobilenet_v1_0.5_160 (both quantized and non-quantized), which can be found [here](https://www.tensorflow.org/lite/guide/hosted_models): 
