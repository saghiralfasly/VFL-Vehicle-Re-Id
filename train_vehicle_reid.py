import keras
import keras.backend as K
import sys
import os
from progressbar import ProgressBar
from keras.callbacks import Callback
import tensorflow as tf
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from scipy.spatial import distance
from keras.models import Model
from PIL import Image
import tensorflow as tf
import random
import os
import numpy
import numpy as np
import time

# defined model
from models.Mob_VFL import load_model  
# dataset utils  
from datasets.vehicle_datasets_utils import prepare_dataset, read_batch


# GPU memory allocation ---------------------------
if 'tensorflow' == K.backend():
  import tensorflow as tf
  from keras.backend.tensorflow_backend import set_session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  set_session(tf.Session(config=config))
  #----------------------------------------------------------

datasets = ['vehicleID','veri']
modes = ["train","resume","inference"]
dataset = 0
mode = 1
img_dim = 224  # tensorflow backend
epochs = 100
batch_size = 12
Augmentation = True

Weights_path = './weights'

n_batches, n_veid = prepare_dataset(datasets[dataset], batch_size, Augmentation)
model, model_name, feature_layer = load_model(input_shape=(img_dim,img_dim,3), n_veid=n_veid, Mode=modes[mode], Weights_path=Weights_path)
print("Model name: " + model_name + "   Target Feature layer: " + feature_layer)

def display_output(output):
  sys.stdout.write("\033[F")
  print(output)
for i in range(epochs):
  current_index = 0
  print("============= Epoch: " + str(i+1))
  print(str(model.metrics_names) + "\n")
  for j in range(n_batches -1):
      images, veid_labels = read_batch(img_dim, j)
      output = model.train_on_batch([images,veid_labels], None)
      display_output(output)
  model.save_weights(os.path.join(weights_folder, model_name))
  print("saved model to : " + os.path.join(weights_folder, model_name) + "\n")



