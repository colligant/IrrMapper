import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import heapq
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import pdb

from scipy.special import expit
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from sys import stdout
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.metrics import Metric
from collections import defaultdict, namedtuple
from multiprocessing import Pool
from random import sample, shuffle
from glob import glob


def mask_unlabeled_values(y_true, y_pred):
    '''
    y_pred: softmaxed tensor
    y_true: one-hot tensor of labels
    Returns two vectors of labels. Assumes input
    tensors are 4-dimensional (batchxrowxcolxdepth)
    '''
    mask = tf.not_equal(tf.reduce_sum(y_true, axis=-1), 0)
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return y_true, y_pred


class StreamingF1Score(Metric):
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
        y_true, y_pred = mask_unlabeled_values(y_true, y_pred)
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        self.cmats.assign_add(tf.cast(tf.math.confusion_matrix(y_true,
           y_pred, num_classes=self.num_classes), tf.float32))
        

    def reset_states(self):
        tf.print(self.cmats, output_stream='file://cmat.out')
        K.batch_set_value([(v, np.zeros((self.num_classes, self.num_classes),
            dtype=np.float32)) for v in self.variables])


    def result(self):
        f1 = self._result(self.cmats)
        if self.focus_on_class is not None:
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


def softmax(arr, count_dim=0):
    arr = np.exp(arr)
    arr /= (np.sum(arr, axis=count_dim, keepdims=True))
    return arr


def make_temporary_directory(model_directory=None):
    if model_directory is None:
        model_directory = './models/'
    temp_dir = os.path.join(model_directory, 'temp') 
    model_path = os.path.join(temp_dir, 'model.h5') 
    tb_path = os.path.join(temp_dir, str(time.time()))
    if not os.path.isdir(tb_path):
        os.makedirs(tb_path)
    return temp_dir, model_path, tb_path


def _bin_dict(dct, k, alpha, n_minority):
    first_edge = min(dct.values())
    last_edge = max(dct.values())

    bin_edges = np.linspace(first_edge, last_edge, k+1, endpoint=True)

    file_dict = defaultdict(list)
    hardness_dict = defaultdict(lambda: 0)

    for data_filename in dct:
        hardness = dct[data_filename]
        for i in range(len(bin_edges)-1):
            if bin_edges[i] <= hardness and hardness < bin_edges[i+1]:
                file_dict[bin_edges[i]].append(data_filename)
                hardness_dict[bin_edges[i]] += hardness
                break # don't need to go on.

    average_hardness_contribution = {}
    for bin_edge in file_dict:
        if not len(file_dict[bin_edge]):
            continue
        average_hardness_contribution[bin_edge] = hardness_dict[bin_edge] / len(file_dict[bin_edge])
    
    sampling_weights = {}
    total_weight = 0
    for bin_edge in average_hardness_contribution:
        t = 1/(alpha + average_hardness_contribution[bin_edge])
        sampling_weights[bin_edge] = t
        total_weight += t

    outfiles = []
    for bin_edge, weight in sampling_weights.items():
        n_samples = int(np.round(weight*n_minority) / total_weight)
        undersample = file_dict[bin_edge]
        if len(undersample) < n_samples:
            undersample *= int(n_samples // len(undersample)) + 1
            # lazy with +1! alternative: add n_samples % len(undersample) files to undersample
        outfiles.extend(sample(undersample, n_samples))
    return outfiles 

def hardbin(negative_example_directory, models, n_minority, alpha, k, custom_objects):
    # Steps:
    # train first model on randomly selected negative examples
    loss_dct = defaultdict(lambda: 0)
    if not isinstance(models, list):
        models = [models]
    print(models)

    files = glob(os.path.join(negative_example_directory, "*.pkl"))
    # parallelize?
    for model_path in models:
        print("Loading model {}".format(model_path))
        model = load_model(model_path, custom_objects=custom_objects)
        for i, f in enumerate(files):
            with open(f, 'rb') as src:
                data = pickle.load(src)
            y_pred = model.predict(np.expand_dims(data['data'], 0))
            mask = data['one_hot'][:, :, 0] == 1 # where there is majority class.
            y_pred = expit(y_pred)
            y_pred = y_pred[0, :, :, 0][mask]
            avg_pred_miss = np.mean(y_pred) #
            # average hardness of tile. A larger number
            # means the network was more sure that the underlying false postive
            # was actually positive.
            loss_dct[f] += avg_pred_miss
        del model

    for f in loss_dct:
        loss_dct[f] /= len(models)

    return _bin_dict(loss_dct, k, alpha, n_minority)


def _preprocess_masks_and_calculate_cmat(y_true, y_pred, n_classes=2):
    labels = range(n_classes)
    if n_classes == 2:
        mask = np.ones_like(y_true).astype(bool)
        mask[y_true == -1] = False
    else:
        mask = np.sum(y_true, axis=2).astype(bool)
    y_pred = y_pred
    if n_classes > 2:
        y_pred = np.squeeze(y_pred)
        y_pred = softmax(y_pred, count_dim=2)
        y_pred = np.argmax(y_pred, axis=2)
        y_true = np.argmax(y_true, axis=2)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    else:
        y_pred = np.round(expit(y_pred))
        y_pred = y_pred[mask]
        y_true = y_true[mask]

    cmat = confusion_matrix(y_true, y_pred,
            labels=labels)

    return cmat

def timeseries_confusion_matrix_from_generator(valid_generator, batch_size, model, n_classes,
        average_time_axis=True):

    out_cmat = np.zeros((n_classes, n_classes))
    labels = np.arange(n_classes)
    if not len(valid_generator):
        raise ValueError("Length of validation generator is 0")

    for count, (batch_x, y_trues) in enumerate(valid_generator):
        preds = np.squeeze(model.predict(batch_x))
        if average_time_axis:
            preds = np.mean(preds, axis=1) # mean , max , last?
            argmax = np.argmax(preds, axis=3) 
            nodata_mask = []
            for i in range(y_trues.shape[0]):
                nodata_mask.append(y_trues[i][0])
            mask = np.asarray(nodata_mask)
            nodata_mask = np.sum(mask, axis=-1) != 0 # no labels in the one hot means sum
            # over depth is 0
            y_hat = argmax[nodata_mask]
            y_true = np.argmax(mask, axis=-1)[nodata_mask]
            out_cmat += confusion_matrix(y_hat, y_true, labels=labels)
            stdout.write("{}/{}\r".format(count, len(valid_generator)))

    return out_cmat, 0, 2 


def confusion_matrix_from_generator(valid_generator, batch_size, model, n_classes=2, print_mat=False, multi_output=False):
    out_cmat = np.zeros((n_classes, n_classes))
    if not len(valid_generator):
        raise ValueError("Length of validation generator is 0")
    with Pool(batch_size) as pool:
        for cnt, (batch_x, y_true) in enumerate(valid_generator):
            y_true = y_true[0] # pull irrigated ground truth
            preds = model.predict(batch_x)
            if multi_output:
                preds = preds[0]
            sz = batch_x[0].shape[0]
            try:
                y_trues = [np.squeeze(y_true[i]) for i in range(sz)]
                y_preds = [np.squeeze(preds[i]) for i in range(sz)]
            except IndexError as e:
                print(e)
                continue
            cmats = pool.starmap(_preprocess_masks_and_calculate_cmat, zip(y_trues, y_preds,
                [n_classes]*batch_size))
            for cmat in cmats:
                out_cmat += cmat
            stdout.write('{}/{}\r'.format(cnt, len(valid_generator)))

        if print_mat:
            print(out_cmat)
        precision_dict = {}
        recall_dict = {}
        for i in range(n_classes):
            precision_dict[i] = 0
            recall_dict[i] = 0
        for i in range(n_classes):
            precision_dict[i] = out_cmat[i, i] / np.sum(out_cmat[i, :]) # row i
            recall_dict[i] = out_cmat[i, i] / np.sum(out_cmat[:, i]) # column i
        return out_cmat, recall_dict, precision_dict


def lr_schedule(epoch, initial_learning_rate, efold):
    lr = initial_learning_rate
    return float(lr*np.exp(-epoch/efold))


def save_model_info(root_directory, loss_func, accuracy, loss, class_weights, classes_to_augment,
        initial_learning_rate, pos_weight, cmat, precision, recall):
    directory_name = os.path.join("./models", "{:.3f}".format(accuracy))
    if os.path.isdir(directory_name):
        directory_name = os.path.join("./models", "{:.5f}acc".format(accuracy))
    filename = os.path.join(directory_name, "run_info_{:.3f}acc.txt".format(accuracy))
    os.rename(root_directory, directory_name)
    print(filename)
    with open(filename, 'w') as f:
        print("acc: {:.3f}".format(accuracy), file=f)
        print("loss_func: {}".format(loss_func), file=f)
        print("loss: {}".format(loss), file=f)
        print("weights: {}".format(class_weights), file=f)
        print("augment scheme: {}".format(classes_to_augment), file=f)
        print("lr: {}".format(initial_learning_rate), file=f)
        print('pos_weight: {}'.format(pos_weight), file=f)
        print('confusion_matrix: {}'.format(cmat), file=f)
        print('precision: {}'.format(precision), file=f)
        print('recall: {}'.format(recall), file=f)


def construct_parser():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-nc", '--n_classes', type=int, default=1)
    parser.add_argument("-p", '--pos-weight', type=float, default=1.0)
    return parser


if __name__ == '__main__':
    pass
