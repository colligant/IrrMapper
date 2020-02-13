import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint, LearningRateScheduler)
from functools import partial
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from scipy.special import expit
from random import sample
from glob import glob
from time import time


from models import unet, two_headed_unet
from data_generators import DataGenerator
from train_utils import lr_schedule, F1Score
from losses import *

join = os.path.join
# don't monitor binary acc any more, monitor precision and recall.
# or monitor top-k accuracy.

if __name__ == '__main__':

    initial_learning_rate = 1e-3

    input_shape = (None, None, 20)

    n_classes = 3

    ap = ArgumentParser()
    ap.add_argument('--gamma', type=float, default=2)

    args = ap.parse_args()

    model = unet(input_shape, initial_exp=4, n_classes=n_classes)
    model_path = 'random_majority_files/multiclass/xen-single-scene-per-season/'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    model_path += 'model.h5'

    pth = '/home/thomas/tensorboard/'+str(time())
    if not os.path.isdir(pth):
        os.mkdir(pth)
    tensorboard = TensorBoard(log_dir=pth,
            profile_batch=0,
            update_freq=30,
            batch_size=3)
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_m_acc',
                                 verbose=1,
                                 save_best_only=True)

    epochs = 1000
    lr_schedule = partial(lr_schedule, initial_learning_rate=initial_learning_rate, efold=200)
    lr_scheduler = LearningRateScheduler(lr_schedule, verbose=True)

    root = '/home/thomas/ssd/single_scene_no_fallow/'
    train_dir = join(root, 'train')
    test_dir = join(root, 'test')

    opt = tf.keras.optimizers.Adam()
    batch_size = 4
    loss_func = masked_categorical_xent
    metric = m_acc
    model.compile(opt, loss=[masked_categorical_xent],
            metrics=[metric])
    train_generator = DataGenerator(train_dir, batch_size, target_classes=None, 
            n_classes=n_classes, balance=False, balance_pixels_per_batch=False, 
            balance_examples_per_batch=True, apply_irrigated_weights=False,
            training=True, augment_data=False, use_cdl=False)
    test_generator = DataGenerator(test_dir, batch_size, target_classes=None, 
            n_classes=n_classes, training=False, balance=False, 
            augment_data=False, use_cdl=False)
    #m2 = F1Score(test_generator, n_classes, model_path, batch_size, two_headed_net=False)
    model.fit_generator(train_generator, 
            epochs=epochs,
            validation_data=test_generator,
            callbacks=[tensorboard, lr_scheduler, checkpoint],#, m2],
            use_multiprocessing=False,
            workers=1,
            max_queue_size=1,
            verbose=1)
