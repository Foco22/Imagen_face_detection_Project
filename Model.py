import cv2
import os
import numpy as np
from PIL import Image
from os import listdir
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_vggface.vggface import VGGFace
from keras.utils.layer_utils import get_source_inputs
import constants as modelo_constants


def my_face(input_shape):

  faces = list()  
  faces_return = list()
  image = Image.open(modelo_constants.picture)
  image = image.convert('RGB')
  pixels = np.array(image)
  detector = MTCNN()
  results = detector.detect_faces(pixels)
  x1, y1, width, height = results[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height
  face = pixels[y1:y2, x1:x2]
  image = Image.fromarray(face)
  image = image.resize(input_shape)
  face_array = np.asarray(image)
  faces.append(face_array)
  faces_return.extend(faces)
  faces_return = np.asarray(faces_return)

  return faces_return


def face_detection(imagePath):

  my_face_picture = my_face((modelo_constants.input_shape_x, modelo_constants.input_shape_y))

  image = Image.open(imagePath)
  image = image.convert('RGB')

  pixels = np.array(image)
  detector = MTCNN()
  results_detection = detector.detect_faces(pixels)

  if len(results_detection) >= 1:
    for faces_imagen in results_detection:
        faces_lista = list()
        face_return = list()

        x1, y1, width, height = faces_imagen['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize((modelo_constants.input_shape_x, modelo_constants.input_shape_y))
        face_array = np.asarray(image)
        faces_lista.append(face_array)
        face_return.extend(faces_lista)
        face_return = np.asarray(face_return)

        model = VGGFace(model='vgg16', include_top=False, input_shape=(modelo_constants.input_shape_x,modelo_constants.input_shape_y,modelo_constants.input_shape_z), pooling='mean')
        
        embedding_original = model.predict(my_face_picture)
        embedding_customer = model.predict(face_return)
        embedding_original = embedding_original.flatten()
        embedding_customer = embedding_customer.flatten()
        
        score = cosine(embedding_original, embedding_customer)
        if score <= modelo_constants.thresh: 
          return 'My face appears in the picture' 

    return 'My face does not appear in the picture'  
  return 'The picture does not contain any faces'