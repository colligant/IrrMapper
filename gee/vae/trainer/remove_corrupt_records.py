import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from train_model import get_dataset, features_dict, to_tuple
from glob import glob


def parse_tfrecord(example_proto):
  """the parsing function.
  read a serialized example into the structure defined by features_dict.
  args:
    example_proto: a serialized example.
  returns:
    a dictionary of tensors, keyed by feature name.
  """
  return tf.io.parse_single_example(example_proto, features_dict)

def make_dataset(root, batch_size=16):

    choice_dataset = tf.data.Dataset.range(len(paths)).repeat()
    dataset = tf.data.experimental.choose_from_datasets(datasets,
            choice_dataset).batch(batch_size).repeat().shuffle(buffer_size=30)
    return dataset
if __name__ == '__main__':
    proto = './data/training_data/test/irrigated/irrigated1590341573.8879178.tfrecord.gz'
    raw_dataset = tf.data.TFRecordDataset([proto], compression_type='GZIP')

    paths = ['irrigated', 'uncultivated', 'unirrigated']
    root = './data/training_data/test/'
    pattern = "*gz"
    datasets = []
    datasets = []
    removed = []
    for path in paths:
        training_root = os.path.join(root, path, pattern)
        for f in glob(training_root):
            dataset = tf.data.TFRecordDataset([f], compression_type='GZIP')
            dataset = dataset.map(parse_tfrecord, num_parallel_calls=5)
            dataset = dataset.map(to_tuple, num_parallel_calls=5)
            try:
                for example in dataset:
                    xx, yy = example[0].shape, example[1].shape
            except Exception as e:
                print(e)
                os.remove(f)
                removed.append(f)
print(len(removed))

