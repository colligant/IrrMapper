import numpy as np
import yaml
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as cbacks
import argparse
import time
import shutil

from types import SimpleNamespace

from . import utils
from . import models


def m_acc(y_true, y_pred):
    y_true_sum = tf.reduce_sum(y_true, axis=-1)
    mask = tf.not_equal(y_true_sum, 0)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    acc = K.mean(K.equal(y_pred_masked, y_true_masked))
    return acc


class StreamingF1Score(tf.keras.metrics.Metric):
    '''
    Stateful metric class to accumulate confusion matrices
    and calculate f1 score from them.
    focus_on_class only reports f1 score for the class labeled with
    the integer that is passed in.
    '''

    def __init__(self, name='f1', num_classes=2, 
            focus_on_class=None, **kwargs):
        super(StreamingF1Score, self).__init__(name=name, **kwargs)
        self.cmats = self.add_weight(name='cmats', shape=(num_classes, 
            num_classes), dtype=tf.float32, initializer='zeros')
        self.num_classes = num_classes
        self.focus_on_class = focus_on_class


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true, y_pred = utils.mask_unlabeled_values(y_true, y_pred)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        self.cmats.assign_add(tf.cast(tf.math.confusion_matrix(y_true,
           y_pred, num_classes=self.num_classes), tf.float32))
        

    def reset_states(self):
        K.batch_set_value([(v, np.zeros((self.num_classes, self.num_classes),
            dtype=np.float32)) for v in self.variables])


    def result(self):
        f1 = self._f1(self.cmats)
        if self.focus_on_class is not None:
            # tf.print(f1[self.focus_on_class], output_stream='file://f1irr.out')
            return f1[self.focus_on_class]
        else:
            return tf.reduce_mean(f1)


    def _f1(self, cmats):
        # returns diagonals of shape (num_classes,).
        prec = cmats / tf.reduce_sum(cmats, axis=1)
        rec = cmats / tf.reduce_sum(cmats, axis=0)
        prec = tf.linalg.tensor_diag_part(prec)
        rec = tf.linalg.tensor_diag_part(rec)
        f1 = 2*(prec*rec)/(rec+prec)
        return f1


def create_new_model_save_directory(model_directory):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return model_directory + '-' +  timestr

class ConfigObj:

    def __init__(self, config_file):

        self.dct = self.dct_from_yaml(config_file)
        self.model_settings = SimpleNamespace(**self.dct['model_settings'])
        self.data_settings = SimpleNamespace(**self.dct['data_settings'])

    def dct_from_yaml(self, yaml_file):

        with open(yaml_file, 'r') as src:
            dct = yaml.safe_load(src)

        return dct

def parse_yaml_config_file(config_file):
   
    return ConfigObj(config_file)

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--job-dir', help='required arg for gcloud')
    ap.add_argument('--config-file', help='YAML config file for hparams')
    args = ap.parse_args()
    config = parse_yaml_config_file(args.config_file)
     
    if config.model_settings.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        if not len(tf.config.list_physical_devices("GPU")):
            print("Could not find GPU, exiting. Change config.yaml\
                        to turn off GPU")
            exit(1)
    
    sf1 = StreamingF1Score(num_classes=config.model_settings.num_classes,
            focus_on_class=config.model_settings.f1_focus_on_class)

    model = models.unet((None, None, config.model_settings.unet_input_channels), 
            n_classes=config.model_settings.num_classes,
            initial_exp=config.model_settings.unet_initial_exp,
            weight_decay_const=config.model_settings.weight_decay_const,
            apply_batchnorm=config.model_settings.apply_batchnorm)


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
           initial_learning_rate=config.model_settings.initial_learning_rate,
           decay_steps=config.model_settings.decay_steps,
           decay_rate=config.model_settings.decay_rate,
           staircase=config.model_settings.staircase)

    optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(optim, loss=config.model_settings.objective, metrics=[m_acc, sf1])

    if config.model_settings.print_model_summary:
        print(model.summary())

    train = utils.make_balanced_training_dataset(os.path.join(config.data_settings.data_root,
        config.data_settings.train_path), 
        batch_size=config.model_settings.batch_size, 
        add_ndvi=config.data_settings.add_ndvi,
        sample_weights=config.data_settings.sample_weights_train,
        year=config.data_settings.train_year,
        buffer_size=config.data_settings.shuffle_buffer_size,
        n_classes=config.model_settings.num_classes
        )

    validation = utils.make_validation_dataset(os.path.join(config.data_settings.data_root, 
        config.data_settings.test_path), 
        batch_size=config.model_settings.batch_size, 
        add_ndvi=config.data_settings.add_ndvi,
        year=config.data_settings.test_year,
        n_classes=config.model_settings.num_classes
        )

    if os.path.isdir(config.data_settings.model_save_directory):
        model_save_directory = os.path.normpath(config.data_settings.model_save_directory) 
        model_save_directory = create_new_model_save_directory(model_save_directory)
        print("Dir. {} already created. Creating new directory {}".format(config.data_settings.model_save_directory, model_save_directory))
        os.mkdir(model_save_directory)
        config.data_settings.model_save_directory = model_save_directory
    else:
        os.mkdir(config.data_settings.model_save_directory)


    shutil.copy2(args.config_file, os.path.join(config.data_settings.model_save_directory, 
        os.path.basename(args.config_file)))
    model_out_path = os.path.join(config.data_settings.model_save_directory, 'model/'
            "/{epoch:03d}_{val_f1:.3f}")
    log_out_path = os.path.join(config.data_settings.model_save_directory,
            'logs/')

    chpt = cbacks.ModelCheckpoint(model_out_path, 
                                  save_best_only=True, 
                                  verbose=True, 
                                  monitor='val_f1',
                                  mode='max') 
                            
    tb = cbacks.TensorBoard(log_dir=log_out_path,
                            update_freq=config.data_settings.tb_update_freq)

    nanloss = cbacks.TerminateOnNaN()

    model.fit(train,
              steps_per_epoch=config.model_settings.training_steps_per_epoch,
              epochs=config.model_settings.epochs,
              validation_data=validation,
              callbacks=[chpt, tb, nanloss],
              verbose=2)


    fully_trained_model_path = os.path.join(log_out_path, "{}_epochs".format(config.EPOCHS))
    model.save(fully_trained_model_path, save_format='tf')
