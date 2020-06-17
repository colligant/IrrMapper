import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import Sequence

from . import  utils
from . import models
from . import feature_spec
from . import config


features_dict = feature_spec.features_dict()
bands = feature_spec.bands()
features = feature_spec.features()

bands = list(features_dict.keys())[:-1]
features = list(features_dict.keys())


def m_acc(y_true, y_pred):
    y_true_sum = tf.reduce_sum(y_true, axis=-1)
    mask = tf.not_equal(y_true_sum, 0)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    acc = K.mean(K.equal(y_pred_masked, y_true_masked))
    return acc


class StreamingF1Score(Metric):
    '''
    Stateful metric class to accumulate confusion matrices
    and calculate f1 score from them.
    focus_on_class only reports f1 score for the class labeled with
    the integer that is passed in.
    '''

    def __init__(self, name='f1', num_classes=2, 
            focus_on_class=None, save_cmat=False, **kwargs):
        super(StreamingF1Score, self).__init__(name=name, **kwargs)
        self.cmats = self.add_weight(name='cmats', shape=(num_classes, 
            num_classes), dtype=tf.float32, initializer='zeros')
        self.num_classes = num_classes
        self.focus_on_class = focus_on_class
        self.save_cmat = save_cmat


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = utils.mask_unlabeled_values(y_true, y_pred)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        self.cmats.assign_add(tf.cast(tf.math.confusion_matrix(y_true,
           y_pred, num_classes=self.num_classes), tf.float32))
        

    def reset_states(self):
        if self.save_cmat:
            tf.print(self.cmats, output_stream='file://cmat.out')
        K.batch_set_value([(v, np.zeros((self.num_classes, self.num_classes),
            dtype=np.float32)) for v in self.variables])


    def result(self):
        f1 = self._result(self.cmats)
        if self.focus_on_class is not None:
            tf.print(f1[self.focus_on_class], output_stream='file://f1irr.out')
            return f1[self.focus_on_class]
        else:
            return tf.reduce_mean(f1)


    def _result(self, cmats):
        # returns diagonals of shape (num_classes,).
        prec = cmats / tf.reduce_sum(cmats, axis=1)
        rec = cmats / tf.reduce_sum(cmats, axis=0)
        prec = tf.linalg.tensor_diag_part(prec)
        rec = tf.linalg.tensor_diag_part(rec)
        f1 = 2*(prec*rec)/(rec+prec)
        return f1



def lr_schedule(epoch):
    lr = 1e-3
    rlr = 1e-3
    if epoch > 25:
        rlr = lr / 2
    if epoch > 50:
        rlr = lr / 4
    if epoch > 75:
        rlr = lr / 6
    if epoch > 100:
        rlr = lr / 8
    if epoch > 125:
        rlr = lr / 16
    if epoch > 150:
        rlr = lr / 32
    tf.summary.scalar('learning rate', data=rlr, step=epoch)
    return rlr

if __name__ == '__main__':


    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir')
    args = ap.parse_args()
    root = args.job_dir


    sf1 = StreamingF1Score(num_classes=config.N_CLASSES, focus_on_class=0)
    model = models.unet((None, None, 30), n_classes=config.N_CLASSES, initial_exp=4)
    model.compile(Adam(1e-3), loss='categorical_crossentropy',
            metrics=[m_acc, sf1])

    train = utils.make_training_dataset(os.path.join('gs://', config.BUCKET, config.DATA_BUCKET, 
        config.TRAIN_BASE), batch_size=config.BATCH_SIZE)
    test = utils.make_test_dataset(os.path.join('gs://', config.BUCKET, config.DATA_BUCKET, 
        config.TEST_BASE), batch_size=config.BATCH_SIZE)

    # train = utils.make_training_dataset('/home/thomas/ssd/train/')
    # test = utils.make_test_dataset('/home/thomas/ssd/test/')

    model_out_path = config.MODEL_DIR
    lr = LearningRateScheduler(lr_schedule, verbose=True)
    chpt = ModelCheckpoint(model_out_path, 
            save_best_only=True, verbose=True, 
            monitor='val_f1', mode='max') 
    
    tb = TensorBoard(log_dir=config.LOGS_DIR,
                     update_freq='epoch')
    model.fit(train,
              steps_per_epoch=config.STEPS_PER_EPOCH,
              epochs=config.EPOCHS,
              validation_data=test,
              validation_steps=config.TEST_SIZE // (2*config.BATCH_SIZE),
              callbacks=[chpt, lr, tb])

    model.save(config.MODEL_DIR, save_format='tf')
