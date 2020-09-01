# safety-kit-detection
This application is designed to detect if people are wearing safety-jackets, hard-hats and a face mask. It's use cuts across different applications such as checking the safety complicance of workers on a site  as to wearing their full safety gear(vest and helment) and also safety compliance of people wearing a face mask as a preventive measure due to the coronavirus or andy other use case of a face mask.

### Demo video of running the app

[![demo video](https://img.youtube.com/vi/jay0QTJs3EI/0.jpg)](https://youtu.be/jay0QTJs3EI)

### Demo picture of mask detection

[![demo picture](https://github.com/mandeelishus/safety-kit-detection/blob/tobe/bin/mask_detection_demo.png)

## Guide for running the app
### Prerequisite
- Openvino (You can run [this script](https://github.com/Tob-iee/OpenVINO_installation) to automate the installation of openvino)

### Which model to use
This application makes use of four models:

- A preson detection model (person-detection-retail-0013 Intel® model), that can be downloaded using the model downloader. The model downloader generates and downloads the IR (the.xml and .bin files) that will be used by the application.
```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "person-detection-retail-0013"
```
- A face detection model (face-detection-adas-binary-0001), that can be downloaded using the model downloader. The model downloader generates and downloads the IR (the .xml and .bin files) that will be used by the application.
```
python /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name "face-detection-adas-binary-0001"
```
- A safety gear detection model (the worker_safety_mobilenet model) , that can be downloaded directly from Openvino's website or from a git project by intel ([link](https://github.com/intel-iot-devkit/safety-gear-detector-python/blob/master/resources/worker-safety-mobilenet)), where it's Caffe* model file are provided. These are then passed through Openvino's model optimizer to generate the IR (the .xml and .bin files) that will be used by the application.
```
python3 /opt/intel/openvino/deployment_tools/tools/model_optimizer/mo_caffe.py --input_model <path to caffemodel file>  --input_proto <path to prototxt file>  -o <specify the output directory>  
```
- A mask detection model (made by DiDi), which can be downloaded from DiDi's mask dietection git repo([link](https://github.com/didi/maskdetection/tree/master/model)), where it's Caffe* model file are provided. These are then passed through Openvino's model optimizer to generate the IR (the .xml and .bin files) that will be used by the application.
```
python3 /opt/intel/openvino/deployment_tools/tools/model_optimizer/mo_caffe.py --input_model <path to caffemodel file>  --input_proto <path to prototxt file>  -o <specify the output directory> 
```
## Directory Structure
 
![Alt text](https://github.com/mandeelishus/safety-kit-detection/blob/tobe/bin/directory_structure.png)

## Demo to run the app
First, initialize the Openvino environment by running the command below
```
source /opt/intel/openvino/bin/setupvars.sh
```
To capture from input path
```
python3 main.py -i <path to the input file>\
-m_g <path to gear detection model IR file> \
-m_p <path to person detection model IR file>\ 
-m_f <path to face detection model IR file> \
-m_m <path to mask detection model IR file> \
-d <specify device> 
```

 ## Documentation
 The required command line arguments are:

-i input_file, which can either be the path of the input video or ```cam``` for camera

-m_g gear_model, path to gear detection model
  
-m_p person_model, path to person detection model
  
-m_f face_model, path to face detection model
  
-m_m mask_model, path to mask detection model
 
The optional command line arguments are:

-l, path for MKLDNN (CPU)-targeted custom layers

-d, target device type e.g. CPU, FPGA

-p, path (in the cloned directory) to store performance statistics i.e. inference time, fps, and model loading time.

-ps, specify detector flags, i.e either (1, 2 or 3) to switch between the gear detector, mask detector or use both respectively


## Observation
When trying to make detection, the angle of the camera to is very important in other to make correct prediction and also an enviroment with good lighting is required for the mask detection model to work properly
