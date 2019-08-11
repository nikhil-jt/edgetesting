mkdir mobilenet_quant
cd mobilenet_quant
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160_quant.tgz
tar -xf mobilenet_v1_0.5_160_quant.tgz
cd ..
echo "deb [trusted=yes] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu
edgetpu_compiler mobilenet_quant/mobilenet_v1_0.5_160_quant.tflite
rm mobilenet_v1_0.5_160_quant_edgetpu.log
mkdir bonnet_compiler
cd bonnet_compiler
wget https://dl.google.com/dl/aiyprojects/vision/bonnet_model_compiler_latest.tgz
tar -xzf bonnet_model_compiler_latest.tgz
cd ..
mkdir mobilenet_float
cd mobilenet_float
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz
tar -xzf mobilenet_v1_0.5_160.tgz
cd ..
./bonnet_compiler/bonnet_model_compiler.par \
  --frozen_graph_path=mobilenet_float/mobilenet_v1_0.5_160_frozen.pb \
  --output_graph_path=mobilenet_vision.binaryproto \
  --input_tensor_name=input \
  --output_tensor_names=MobilenetV1/Predictions/Softmax \
  --input_tensor_size=160 \
  --debug
cp mobilenet_quant/mobilenet_v1_0.5_160_quant.tflite mobilenet_jetson.tflite
mv mobilenet_v1_0.5_160_quant_edgetpu.tflite mobilenet_coral.tflite
sudo rm -r bonnet_compiler
sudo rm -r mobilenet_float
sudo rm -r mobilenet_quant
