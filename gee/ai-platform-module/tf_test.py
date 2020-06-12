import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt 
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Metric
from tensorflow.keras.utils import Sequence
from test import features_dict
from models import unet

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

N_CLASSES = 3 

bands = list(features_dict.keys())[:-1]
features = list(features_dict.keys())

def parse_tfrecord(example_proto):
  """the parsing function.
  read a serialized example into the structure defined by features_dict.
  args:
    example_proto: a serialized example.
  returns:
    a dictionary of tensors, keyed by feature name.
  """
  return tf.io.parse_single_example(example_proto, features_dict)

def one_hot(labels):
    h, w, d = labels.shape
    labels = tf.squeeze(labels)
    ls = []
    for i in range(N_CLASSES):
        if i == 0:
            where = tf.where(labels != i+1, tf.zeros((h, w)), 1*tf.ones((h,w)))
        else:
            where = tf.where(labels != i+1, tf.zeros((h, w)), tf.ones((h,w)))
        ls.append(where)
    temp = tf.stack(ls, axis=-1)
    return temp


def to_tuple(inputs):
  """Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
  Turn the tensors returned by parse_tfrecord into a stack in HWC shape.
  Args:
    inputs: A dictionary of tensors, keyed by feature name.
  Returns:
    A tuple of (inputs, outputs).
  """
  inputsList = [inputs.get(key) for key in sorted(features)]
  stacked = tf.stack(inputsList, axis=0)
  # Convert from CHW to HWC
  stacked = tf.transpose(stacked, [1, 2, 0])
  inputs = stacked[:,:,:len(bands)] 
  labels = one_hot(stacked[:,:,len(bands):])
  labels = tf.cast(labels, tf.int32)
  return inputs, labels


def get_dataset(pattern):
  """Function to read, parse and format to tuple a set of input tfrecord files.
  Get all the files matching the pattern, parse and convert to tuple.
  Args:
    pattern: A file pattern to match in a Cloud Storage bucket.
  Returns:
    A tf.data.Dataset
  """
  glob = tf.io.gfile.glob(pattern)
  dataset = tf.data.TFRecordDataset(glob, compression_type='GZIP')
  dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
  dataset = dataset.map(to_tuple, num_parallel_calls=5)
  return dataset

def m_acc_class_0(y_true, y_pred):
    y_true_sum = tf.reduce_sum(y_true, axis=-1)
    mask = tf.not_equal(y_true_sum, 0)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    mask = tf.equal(y_true_masked, 0)
    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)
    acc = K.mean(K.equal(y_pred_masked, y_true_masked))
    return acc

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

def plot_balance(dataset, steps):

    stp = []
    for i, batch in enumerate(dataset):
        labels = batch[1]
        for j in range(labels.shape[0]):
            label = labels[j].numpy()
            good = np.where(np.sum(label, axis=-1) != 0)
            label = np.argmax(label,axis=-1)[good]
            unique, counts = np.unique(label, return_counts=True)
            print(unique, counts)
        if i > steps:
            break

class DataGenerator(Sequence):

    def __init__(self, datasets, min_length,
            batch_size):

        for dataset in datasets:
            dataset = dataset.repeat().batch(batch_size).shuffle(buffer_size=30)
        self.datasets = datasets
        self.min_length = min_length
        self.subset_features = []
        self.subset_labels = []
        self.count = 0

    def __len__(self):
        return self.min_length

    def __getitem__(self, idx):

        if self.count < 10:
            labels = []
            features = []
            for i in datasets:
                instance = i.skip(idx)
                instance = list(instance.take(1).as_numpy_iterator())[0]
                labels.append(instance[1])
                features.append(instance[0])
            labels = tf.stack(labels, axis=0)
            features = tf.stack(features, axis=0)
            self.subset_features.append(features)
            self.subset_labels.append(labels)
            self.count += 1
            return features, labels, [None]
        else:
            return self.subset_features[idx % 2], self.subset_labels[idx%2], [None]


if __name__ == '__main__':
    pattern = "*gz"
    paths = ['irrigated', 'uncultivated', 'unirrigated']
    root = './training_data/'

    sf1 = StreamingF1Score(num_classes=3, focus_on_class=0)
    model = unet((None, None, 42), n_classes=N_CLASSES, initial_exp=4)
    model.compile(Adam(1e-4), loss='categorical_crossentropy',
            metrics=[m_acc, sf1])

    datasets = []
    i = 0
    for path in paths:
        training_root = os.path.join(root, path, pattern)
        dataset = get_dataset(training_root)
        datasets.append(dataset.repeat())

    # gen = DataGenerator(datasets, 300, batch_size=4)

    choice_dataset = tf.data.Dataset.range(3).repeat()
    total = tf.data.experimental.choose_from_datasets(datasets,
            choice_dataset).batch(16).repeat().shuffle(buffer_size=30)

    # for batch in total:
    #     batch = batch[1]
    #     instances_per_batch = [0, 0, 0]
    #     for i in range(batch.shape[0]):
    #         slc = batch[i].numpy()
    #         good = np.where(np.sum(slc, axis=-1) != 0)
    #         slc = np.argmax(slc, axis=-1)
    #         slc = slc[good]
    #         unique = np.unique(slc)
    #         instances_per_batch[unique[-1]] += 1
    #     print(instances_per_batch)

    model.fit(total, verbose=True,
              steps_per_epoch=300,
              epochs=100)

    total = total.unbatch().batch(1)
    for batch in total:
        preds = model.predict(batch)
        for i in range(preds.shape[0]):
            pred = preds[i]
            label = batch[1][i].numpy().squeeze()
            print(label.shape, pred.shape)
            print(m_acc(label, pred))
            fig, ax = plt.subplots(ncols=3)
            bad = np.where(np.sum(label, axis=-1) == 0)
            label = label.astype(np.float32)
            label[bad] = np.nan
            ax[0].imshow(pred)
            ax[1].imshow(label)
            ax[2].imshow(batch[0][i, :, :, :3])
            plt.show()
