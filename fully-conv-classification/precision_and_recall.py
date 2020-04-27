import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
from tensorflow.keras.models import load_model
from glob import glob
from numpy import sum as nsum

from losses import *
from models import *
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

    model = unet((None, None, 98), n_classes=3, initial_exp=5)
    base = './current_models/non-recurrent/full-unet-random_start_date-diff-lr-with-centroids/'
    ### BEST MODEL: but precision is 89.2 and recall is 0.94
    model_path = base + 'model_0.973-0.917.h5'
    ### BEST MODEL k
    model.load_weights(model_path)
    batch_size = 8
    min_images = 14
    test_data_path = '/home/thomas/ssd/training-data/training-data-l8-centroid-all-bands-full-year-16-img/test/'
    n_classes = 3
    test_generator = StackDataGenerator(data_directory=test_data_path, batch_size=batch_size,
            training=False, min_images=min_images)
    print(len(test_generator))
    cmat, prec, recall = confusion_matrix_from_generator(test_generator, batch_size, 
            model, n_classes=n_classes, time_dependent=False)
    #cmat, prec, recall = timeseries_confusion_matrix_from_generator(test_generator, batch_size, 
    #        model, n_classes=args.n_classes)
    print(nsum(cmat, axis=1))
    print('\n p:{}\n r:{}'.format(prec, recall))
