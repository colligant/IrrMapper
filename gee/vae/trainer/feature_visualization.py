import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import gc
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
import numpy as np
from pdb import set_trace
from glob import glob
from sys import stdout, exit
from rasterio import open as rasopen
import rasterio
from tensorflow.keras.models import load_model
from models import unet
from random import shuffle
import feature_spec

base = './models/'
model_path = './../../gs-models/0.8820/'
loaded = tf.saved_model.load(model_path)
model = unet((None, None, 30), initial_exp=4)

trained_vars = loaded.variables

for trained_weight, weight in zip(trained_vars, model.weights):
    print(trained_weight.name, weight.name)

