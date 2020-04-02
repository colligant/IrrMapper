import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tensorflow.keras.models import load_model
from glob import glob
from numpy import sum as nsum

from losses import *
from models import small_unet_smarter
from data_generators import StackDataGenerator
from train_utils import confusion_matrix_from_generator, timeseries_confusion_matrix_from_generator


if __name__ ==  '__main__':

    #import numpy as np
    #arr = np.array([[8.3913000e+04,1.8664000e+04,1.0272000e+04],
    #                 [1.6963000e+04,1.1221567e+07,7.2342400e+05],
    #                 [1.1794000e+04,1.7684930e+06,3.9401140e+06]])
    #n_classes = 3
    #precision_dict = {}
    #recall_dict = {}
    #for i in range(n_classes):
    #    precision_dict[i] = 0
    #    recall_dict[i] = 0
    #for i in range(n_classes):
    #    precision_dict[i] = arr[i, i] / np.sum(arr[i, :]) # row i
    #    recall_dict[i] = arr[i, i] / np.sum(arr[:, i]) # column i
    #print(precision_dict)
    #print(recall_dict)

    model = small_unet_smarter((None, None, 39), base=6)
    model_path = './recurrent_0.878.h5'
    model = tf.keras.models.load_model(model_path, custom_objects={'m_acc':m_acc})
    batch_size = 8
    test_data_path = '/home/thomas/ssd/training-data-l8-no-centroid-full-year/test/'
    n_classes = 3

    test_generator = StackDataGenerator(data_directory=test_data_path, batch_size=batch_size,
            training=False, min_rgb_images=13)
    print(len(test_generator))
    cmat, prec, recall = confusion_matrix_from_generator(test_generator, batch_size, 
            model, n_classes=n_classes, time_dependent=True)
    #cmat, prec, recall = timeseries_confusion_matrix_from_generator(test_generator, batch_size, 
    #        model, n_classes=args.n_classes)
    print(cmat)
    print(nsum(cmat, axis=1))
    print('\n p:{}\n r:{}'.format(prec, recall))
