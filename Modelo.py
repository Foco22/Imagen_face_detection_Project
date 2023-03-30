import requests
import cv2
import os
import pickle
import numpy as np
from PIL import Image
from os import listdir
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from keras_vggface.vggface import VGGFace
from keras.utils.layer_utils import get_source_inputs

required_size=(224, 224)  
thresh=0.5

########################## MODELO #####################################

def my_face():


  filename_list = ['francisco.jpeg']
     

  picture_list = []
  for picture in filename_list:
    faces = list()
    X = list()
    image = Image.open(picture)

    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)



    face_array = np.asarray(image)
    faces.append(face_array)
    X.extend(faces)
    X = np.asarray(X)

    picture_list.append(X)

    return X

def detecion_face(imagePath):

  #image = cv2.imread(imagePath,cv2.IMREAD_COLOR)[:,:,::-1]
  my_face_picture = my_face()

  image = Image.open(imagePath)
  #image = image.convert('RGB')
  pixels = np.asarray(image)
  detector = MTCNN()
  results = detector.detect_faces(pixels)

  if len(results) >= 1:
    for faces_r in results:
        faces_lista = list()
        X = list()

        x1, y1, width, height = faces_r['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)

        face_array = np.asarray(image)
        faces_lista.append(face_array)
        X.extend(faces_lista)
        X = np.asarray(X)

        model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        
        embedding_original = model.predict(my_face_picture)
        embedding_customer = model.predict(X)
        
        embedding_original = embedding_original.flatten()
        embedding_customer = embedding_customer.flatten()
        
        score = cosine(embedding_original, embedding_customer)
        if score <= thresh: 
          #cv2.rectangle(img_reactangulo, (x1, y1), (x2, y2), (255, 0, 0), 2)  
          #result_BGR = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
          #cv2.imshow("Cropped Face", face)
          #cv2.imwrite('face.jpg', result_BGR)
          return 'My face appears in the picture'

    return 'My face does not appear in the picture'  
  return 'The picture does not contain any faces'


#print(detecion_face('francisco1.jpeg'))

