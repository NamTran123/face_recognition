from face_detect import  load_faces , load_dataset
from  Face_Embeddings import faces_embeddings
from numpy import savez_compressed 
from face_classfication_model import train_SVM
from numpy import load
import numpy as  np  

#load train dataset
trainX, trainy = load_dataset('train/')
print(trainX.shape, trainy.shape)
# load test dataset
testX, testy = load_dataset('val/')
# save arrays to one file in compressed format

savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)

# load the face dataset
data = load('5-celebrity-faces-dataset.npz')

faces_embeddings(data)

# load dataset
data = load('5-celebrity-faces-embeddings.npz')
train_SVM(data)
