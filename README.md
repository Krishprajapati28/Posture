# Age Prediction Model

A Python application that predicts age from facial images using pre-trained deep learning models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download pre-trained models:

Download these files and place them in the same directory:

- Face Detection:
  - [opencv_face_detector.pbtxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/opencv_face_detector.pbtxt)
  - [opencv_face_detector_uint8.pb](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb)

- Age Prediction:
  - [age_deploy.prototxt](https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/age_net_definitions/age_deploy.prototxt)
  - [age_net.caffemodel](https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel)

## Usage

### Predict age from image:
```bash
python age_prediction.py path/to/image.jpg
```

### Real-time webcam prediction:
```bash
python age_prediction.py
```

Press 'q' to quit webcam mode.

## Features

- Detects multiple faces in an image
- Predicts age ranges: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- Real-time webcam support
- Saves annotated output images

## Model Information

Uses pre-trained Caffe models from OpenCV and the Age & Gender Deep Learning project.
