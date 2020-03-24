# LD_LIBRARY_PATH="/usr/local/cuda-10.0/extras/CUPTI/lib64" <- gotta run for TB to work.
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
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
from lstm_conv_model import model as m_model


class StreamingConfusionMatrix(Metric):

    def __init__(self, name='conf_mat', out_fname='test.txt', num_classes=2, **kwargs):
        super(StreamingConfusionMatrix, self).__init__(name=name, **kwargs)
        self.cmats = self.add_weight(name='cmats', shape=(num_classes, 
            num_classes), dtype=tf.float32, initializer='zeros')
        self.num_classes = num_classes
        self.out_fname = out_fname

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.num_classes < 3:
            y_pred = tf.round(y_pred)
        else:
            y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        self.cmats.assign_add(tf.cast(tf.math.confusion_matrix(y_true,
           y_pred, num_classes=self.num_classes), tf.float32))
        

    def reset_states(self):
        tf.print(self.cmats, output_stream='file://cmat.out')
        K.batch_set_value([(v, np.zeros((self.num_classes, self.num_classes),
            dtype=np.float32)) for v in self.variables])


    def result(self):
        inte = self._result(self.cmats)
        tf.print(self.cmats)
        tf.print(inte)
        return inte

    def _result(self, cmats):
        # precision for minority class,...
        prec = cmats / tf.reduce_sum(cmats, axis=1)
        rec = cmats / tf.reduce_sum(cmats, axis=0)
        prec = tf.linalg.tensor_diag_part(prec)
        rec = tf.linalg.tensor_diag_part(rec)
        return (prec, rec)


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

    #model = small_lstm_unet((None, None, None, 3), base_filters=3)
    # model = small_unet_residual_connections((None, None, 39))
    model = small_unet_smarter((None, None, 39))
    # model.summary()

    train_path = '/home/thomas/ssd/training-data-l8-no-centroid/train'
    test_path = '/home/thomas/ssd/training-data-l8-no-centroid/test'

    train_generator = StackDataGenerator(train_path, 16, 
            only_irrigated=False)
    test_generator = StackDataGenerator(test_path, 32, 
            only_irrigated=False, training=False)

    model.compile(Adam(1e-3), loss='categorical_crossentropy',
            metrics=[m_acc])

    model.summary()

    model_name = 'model_{val_m_acc:.3f}.h5'
    model_dir = 'newlrschedule'
    model_out_path = 'models/non-recurrent/{}/'.format(model_dir)


    if not os.path.isdir(model_out_path):
        os.makedirs(model_out_path)

    model_out_path += model_name
    print(model_out_path)

    lr = LearningRateScheduler(lr_schedule, verbose=True)

    if not os.path.isfile(model_out_path):
        chpt = ModelCheckpoint(model_out_path, save_best_only=True, verbose=True,
                monitor='val_m_acc')
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
        model = tf.keras.models.load_model(model_out_path, custom_objects={'m_acc':m_acc,
            'masked_categorical_xent':masked_categorical_xent})
    valid_generator = SeriesDataGenerator(train_path, 1, training=True,
            only_irrigated=False)
    for i, (image, mask) in enumerate(valid_generator):
        # image = image[:, :, :, :]
        preds = np.squeeze(model.predict(image))
        preds = preds / np.max(preds)
        image = np.squeeze(image)
        mask = np.squeeze(mask)
        mask = mask.astype(np.float)
        image = image.astype(np.float)
        image = _norm(image)
        preds = _norm(preds)
        idx = image.shape[0]-1
        for idx in range(image.shape[0]):
            fig, ax = plt.subplots(ncols=3)
            ax[0].imshow(image[idx])
            ax[1].imshow(mask[idx])
            ax[2].imshow(preds[idx])
            plt.suptitle(idx)
            plt.show()
