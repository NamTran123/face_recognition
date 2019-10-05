# Multiple Face Recogniton using FACENET and  SVM
OpenCV based face recognition system that can detect and recognize multiple faces in an image. I have implemented a multiple face recognition system.



There are 2 parts in a face recognition system.
  1. Face_Embeddings  - To extract  features  from  images.
  2. Face Recognition - To recognize face of  persons in the images using  SVM.
  
## 1. Face_Embeddings.
Face embeddings using facenet .

## 2. Face Recognition
I using SVM .

## Requirements
1. [Python 3.6.x](https://www.python.org/downloads/)
2. [OpenCV 3](https://opencv.org/releases/)
3. [Numpy](https://www.numpy.org/)
4. All in   requiments.txt

## How to run?
`sudo pip3 install -r requiments.txt`

`python3 main.py`

The photos of each individual should be stored in a folder train inside the `train` folder .
Test images are stored in `test` folder.

The application is built over 3 files. 
  1. data_processing.py - Load  file and  Create train ,test   
  2. face_embeddings  - Detect face and extract feature face embeddings using facenet.
  3. face_classification_model.py - To  classification  face  for face recognition  using svm .
