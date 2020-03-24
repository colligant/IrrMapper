import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from sys import stdout
from cv2 import resize, imwrite
from glob import glob
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D,
        UpSampling2D, Concatenate, Add)
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix
from random import shuffle

from data_generators import SeriesDataGenerator, StackDataGenerator
from losses import m_acc, masked_categorical_xent
from models import *
from train_utils import StreamingF1Score


def _norm(im):
    return im / np.max(im)


def lr_schedule(epoch):
    lr = 1e-3
    rlr = 1e-3
    if epoch > 100:
        rlr = lr / 2
    if epoch > 200:
        rlr = lr / 4
    if epoch > 350:
        rlr = lr / 6
    if epoch > 400:
        rlr = lr / 8
    if epoch > 500:
        rlr = lr / 16
    print('Learning rate: ', rlr)
    return rlr

if __name__ == '__main__':

    model = small_unet_smarter((None, None, 39))

    train_path = '/home/thomas/ssd/training-data-l8-no-centroid/train'
    test_path = '/home/thomas/ssd/training-data-l8-no-centroid/test'

    train_generator = StackDataGenerator(train_path, 2, 
            only_irrigated=False)

    test_generator = StackDataGenerator(test_path, 32, 
            only_irrigated=False, training=False)

    sf1 = StreamingF1Score(num_classes=3, focus_on_class=0)

    model.compile(Adam(1e-3), loss='categorical_crossentropy',
            metrics=[m_acc, sf1])

 #   model.summary()

    model_name = 'model_{val_m_acc:.3f}-{val_f1:.3f}.h5'
    model_dir = 'recording-f1'
    model_out_path = 'current_models/non-recurrent/{}/'.format(model_dir)


    if not os.path.isdir(model_out_path):
        os.makedirs(model_out_path)

    model_out_path += model_name

    # Callbacks
    lr = LearningRateScheduler(lr_schedule, verbose=True)
    chpt = ModelCheckpoint(model_out_path, 
            save_best_only=True, verbose=True, 
            monitor='val_f1') 

    if not os.path.isfile(model_out_path):
        model.fit_generator(train_generator,
                epochs=1000,
                validation_data=test_generator,
                validation_freq=1,
                callbacks=[chpt, lr],
                use_multiprocessing=False,
                workers=1,
                verbose=True)
        model.save('full.h5')
    else:
        model.load_weights(model_out_path)
