from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import ImageDraw, ImageEnhance
import keras.backend as K
import sys
import os
import tensorflow as tf
import random
import glob
import math
import numpy
from PIL import Image

datset_base_folder = ' '
datset_folder_test_veri = '/media/saghir/Data_Part/datasets/veri/image_test'
datset_folder_test_vehicleID = '/media/saghir/Data_Part/datasets/veID/image'

Augmentation = True
aug_crop = True
aug_bright = True
aug_flip = False
aug_rotate = True


dataset_to_read = ''
# batch_size = 0
num_batches = 0
n_colors = 0
n_types = 0
n_veid = 0
n_viewp = 0
n_classes = 576
training_lines = []
query_lines = []
test_lines = []

def prepare_dataset(dataset,b_size,Augment):
  global dataset_to_read
  dataset_to_read = dataset
  global batch_size
  Augmentation = Augment
  global training_lines
  global n_veid
  global datset_base_folder
  batch_size = b_size

  if dataset_to_read == 'vehicleID':
    n_veid = 13164
    datset_base_folder = '/media/saghir/Data_Part/datasets/veID/image'
    train_file = open('/media/saghir/Data_Part/datasets/veID/train_test_split/train_list_modified_13134.txt', 'r')
  else: 
    n_veid = 576
    # TODO, update the path of the training part images
    datset_base_folder = '/media/saghir/Data_Part/datasets/veri/image_train'
    train_file = open('/media/saghir/Data_Part/datasets/veri/final_train_list_modified.txt', 'r')
  train_lines = train_file.readlines()
  for line in train_lines:
        training_lines.append(line.rstrip().split(' '))

  n_batches = math.floor(len(training_lines) / batch_size)
  perm = list(range(len(training_lines)))
  random.shuffle(perm)
  training_lines = [training_lines[index] for index in perm]
  data_trining_size = len(training_lines)
  print("Dataset " + str(dataset_to_read) + " is ready with " + str(data_trining_size)
            + " images for training and " + str(n_veid) + " vehicle IDs .....")
  return(n_batches, n_veid)

def read_batch(input_dim, b_n):
    index = 0
    B = numpy.zeros(shape=(batch_size, input_dim, input_dim, 3))
    L_veID = numpy.zeros(shape=(batch_size,n_veid))
    data_index = b_n * batch_size
    while index < batch_size:
            img = load_img(os.path.join(datset_base_folder, training_lines[data_index][0]))
            if Augmentation:
               img = augment_image(img)
            img1 = img.resize((input_dim,input_dim),Image.ANTIALIAS)
            B[index] = img_to_array(img1)
            if dataset_to_read == 'veri':
                L_veID[index] = keras.utils.to_categorical(training_lines[data_index][4], n_veid)
            else:
                L_veID[index] = keras.utils.to_categorical(training_lines[data_index][1], n_veid)
            index += 1
            data_index += 1
    B = B[..., ::-1]
    # Zero-center by mean pixel
    if dataset_to_read == 'veri':
       B[..., 0] -= 107.223  
       B[..., 1] -= 107.137  
       B[..., 2] -= 108.704  
    else:
        B[..., 0] -= 98.9369  
        B[..., 1] -= 103.911  
        B[..., 2] -= 104.18  
    B *= 0.017 # scale values
    return B, L_veID

def augment_image(im):
  img = im
  if aug_crop:
    w, h = im.size
    wc = random.randint((w//1.1),w)
    hc = random.randint((h//1.1),h)
    xc = random.randint(0,(w-wc))
    yc = random.randint(0,(h-hc))
    img = im.crop((xc, yc, xc+wc, yc+hc))
  if aug_bright:
    im_br = ImageEnhance.Brightness(img)
    img = im_br.enhance(random.uniform(0.8, 1.2))
  if aug_rotate:
    img = img.rotate(random.uniform(-2.0, 2.0), expand=False)
  return img

