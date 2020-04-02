import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import keras

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
import tensorflow.keras.metrics as km
from sklearn.metrics import confusion_matrix
from random import shuffle

from data_generators import SeriesDataGenerator, StackDataGenerator
from losses import m_acc, masked_categorical_xent
from models import *
from train_utils import StreamingF1Score

km.StreamingF1Score = StreamingF1Score

def _norm(im):
    return im / np.max(im)


def lr_schedule(epoch):
    lr = 1e-3
    rlr = 1e-3
    if epoch > 50:
        rlr = lr / 2
    if epoch > 100:
        rlr = lr / 4
    if epoch > 150:
        rlr = lr / 6
    if epoch > 200:
        rlr = lr / 8
    if epoch > 250:
        rlr = lr / 16
    tf.summary.scalar('learning rate', data=rlr, step=epoch)
    return rlr

if __name__ == '__main__':

    # model = unet((None, None, 39), n_classes=3, initial_exp=5)
    model = small_unet_smarter((None, None, 39), base=4)

    train_path = '/home/thomas/ssd/training-data-l8-no-centroid-full-year/train'
    test_path = '/home/thomas/ssd/training-data-l8-no-centroid-full-year/test'

    train_generator = StackDataGenerator(train_path, 32, 
            only_irrigated=False, min_rgb_images=13)

    test_generator = StackDataGenerator(test_path, 64, 
            only_irrigated=False, training=False, min_rgb_images=13)

    sf1 = StreamingF1Score(num_classes=3, focus_on_class=0)

    model.compile(Adam(1e-3), loss='categorical_crossentropy',
            metrics=[m_acc, sf1])

    model.summary()

    model_name = 'model_{val_m_acc:.3f}-{val_f1:.3f}.h5'
    model_dir = 'small-unet-full-year'
    model_out_path = 'current_models/non-recurrent/{}/'.format(model_dir)
    tb_path = os.path.join(model_out_path, 'logs')

    if not os.path.isdir(model_out_path):
        os.makedirs(model_out_path)
    if not os.path.isdir(tb_path):
        os.makedirs(tb_path)

    model_out_path += model_name

    # Callbacks
    file_writer = tf.summary.create_file_writer(tb_path + "/metrics")
    file_writer.set_as_default()
    lr = LearningRateScheduler(lr_schedule, verbose=True)
    chpt = ModelCheckpoint(model_out_path, 
            save_best_only=True, verbose=True, 
            monitor='val_f1', mode='max') 

    print(tb_path)
    tb = TensorBoard(log_dir=tb_path,
                     update_freq='epoch',
                     write_images=True,
                     histogram_freq=3)

    if not os.path.isfile(model_out_path):
        model.fit_generator(train_generator,
                epochs=1000,
                validation_data=test_generator,
                validation_freq=1,
                callbacks=[chpt, lr, tb],
                use_multiprocessing=False,
                workers=1,
                verbose=True)
        model.save('full.h5')
    else:
        model.load_weights(model_out_path)
