import numpy as np
import time
import os
import tensorflow as tf
import tensorflow
from collections import defaultdict
from random import shuffle

from . import feature_spec

features_dict = feature_spec.features_dict()
BANDS = feature_spec.bands() # includes mask raster
FEATURES = feature_spec.features() # only input features
NDVI_INDICES = [(2, 3), (8, 9), (14, 15), (20, 21), (26, 27), (32, 33)]

def tf_distance_map(mask):
    im_shape = mask.shape
    mask = tf.cast(mask, tf.bool)
    mask = tf.math.logical_not(mask) # make the non-masked areas masked
    [mask,] = tf.py_function(distance_map, [mask], [tf.float32])
    mask.set_shape(im_shape)
    return mask

def distance_map(mask):
    mask = distance_transform_edt(mask) 
    mask[0, 0] = 0
    return mask

def one_hot(labels, n_classes):
    h, w = labels.shape
    ls = []
    for i in range(n_classes):
        ls.append(tf.where(labels == i+1, tf.ones((h, w)), tf.zeros((h, w))))
    return tf.stack(ls, axis=-1)


def one_hot_border_labels(labels, n_classes):
    h, w, d = labels.shape
    labels = tf.squeeze(labels)
    ls = []
    border_labels = None
    for i in range(n_classes):
        if i == 0:
            # informative names, here
            where = tf.where(labels != i+1, tf.zeros((h, w)), 1*tf.ones((h,w)))
            border_labels = tf_distance_map(where)
            border_labels = tf.where(border_labels != i+1, tf.zeros((h, w)), tf.ones((h,w)))
        elif i == 2 and border_labels is not None:
            where = tf.where(labels != i+1, tf.zeros((h, w)), tf.ones((h,w)))
            where = tf.where(border_labels == 1, 10*tf.ones((h, w)), where)
        else:
            where = tf.where(labels != i+1, tf.zeros((h, w)), tf.ones((h,w)))
        ls.append(where)
    temp = tf.stack(ls, axis=-1)
    return temp

def mask_unlabeled_values(y_true, y_pred):
    '''
    y_pred: softmaxed tensor
    y_true: one-hot tensor of labels
    Returns two vectors of labels. Assumes input
    tensors are 4-dimensional (batchxrowxcolxdepth),
    and nodata is indicated by all 0s over depth in
    the one-hot tensor.
    '''
    mask = tf.not_equal(tf.reduce_sum(y_true, axis=-1), 0)
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return y_true, y_pred


def confusion_matrix_from_generator(datasets, batch_size, model, n_classes):
    ''' 
    inputs: list of tf.data.Datasets, not batched, without repeat.
    '''   
    out_cmat = np.zeros((n_classes, n_classes))
    labels = range(n_classes)
    instance_count = 0
    for dataset in datasets:
        for batch in dataset:
            features, y_true = batch[0], batch[1]
            y_pred = model(features)['softmax'].numpy()
            instance_count += y_pred.shape[0]
            y_true, y_pred = mask_unlabeled_values(y_true, y_pred)
            cmat = tf.math.confusion_matrix(y_true, y_pred, num_classes=n_classes)
            out_cmat += cmat
    precision_dict = {}
    recall_dict = {}
    for i in range(n_classes):
        precision_dict[i] = 0
        recall_dict[i] = 0
    for i in range(n_classes):
        precision_dict[i] = out_cmat[i, i] / np.sum(out_cmat[i, :]) 
        recall_dict[i] = out_cmat[i, i] / np.sum(out_cmat[:, i]) 
    return out_cmat, recall_dict, precision_dict, instance_count


def get_shared_dataset(pattern, add_ndvi):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
      pattern: A file pattern to match in a Cloud Storage bucket,
               or list of GCS files
      add_ndvi:  Whether or not to add ndvi to the feature stak, computed on the fly.

      Returns:
      A tf.data.Dataset
    """
    if not isinstance(pattern, list):
        pattern = tf.io.gfile.glob(pattern)
    shuffle(pattern)
    dataset = tf.data.TFRecordDataset(pattern, compression_type='GZIP',
            num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    to_tup = to_shared_tuple(add_ndvi)
    dataset = dataset.map(to_tup, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def get_dataset(pattern, add_ndvi, n_classes):
    """Function to read, parse and format to tuple a set of input tfrecord files.
    Get all the files matching the pattern, parse and convert to tuple.
    Args:
      pattern: A file pattern to match in a Cloud Storage bucket,
               or list of GCS files
      add_ndvi:  Whether or not to add ndvi to the feature stak, computed on the fly.
      n_classes: The number of classes in the segmentation dataset, used 
                 to define the shape of the one hot matrix.
    Returns:
      A tf.data.Dataset
    """
    if not isinstance(pattern, list):
        files = tf.io.gfile.glob(pattern)
        shuffle(files)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP',
                num_parallel_reads=tf.data.experimental.AUTOTUNE)
    else:
        shuffle(pattern)
        dataset = tf.data.TFRecordDataset(pattern, compression_type='GZIP',
                num_parallel_reads=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    to_tuple_fn = to_tuple(add_ndvi, n_classes)
    dataset = dataset.map(to_tuple_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def parse_tfrecord(example_proto):
    """the parsing function.
    read a serialized example into the structure defined by features_dict.
    args:
      example_proto: a serialized example.
    returns:
      a dictionary of tensors, keyed by feature name.
    """
    return tf.io.parse_single_example(example_proto, features_dict)

def to_shared_tuple(add_ndvi):
    """
    Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.  
    Args: inputs: A dictionary of tensors, keyed by feature name.
    Returns:
      A tuple of ((inputs_0, ..., inputs_n), outputs), where inputs is
      slices of the input tensor with channel dimension 6.
    """
    def _to_tuple(inputs):
        features_list = [inputs.get(key) for key in sorted(FEATURES)]
        stacked = tf.stack(features_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0]) * 0.0001
        out = []
        for i in range(6, stacked.shape[-1], 6):
            out.append(stacked[:, :, i-6:i])
        # 'constant' is the label for label raster. 
        labels = one_hot(inputs.get('constant'), n_classes=3)
        return (out[0], out[1], out[2], out[3], out[4], out[5]), labels

    return _to_tuple

def to_tuple(add_ndvi, n_classes):
    """
    Function to convert a dictionary of tensors to a tuple of (inputs, outputs).
    Turn the tensors returned by parse_tfrecord into a stack in HWC shape.  
    Args: inputs: A dictionary of tensors, keyed by feature name.
    Returns:
      A tuple of (inputs, outputs).
    """
    def _to_tuple(inputs):
        features_list = [inputs.get(key) for key in sorted(FEATURES)]
        stacked = tf.stack(features_list, axis=0)
        # Convert from CHW to HWC
        stacked = tf.transpose(stacked, [1, 2, 0]) * 0.0001
        if add_ndvi:
            image_stack = add_ndvi_raster(stacked)
        else:
            image_stack = stacked
        # 'constant' is the label for label raster. 
        labels = one_hot(inputs.get('constant'), n_classes=n_classes)
        return image_stack, labels

    return _to_tuple

def add_ndvi_raster(image_stack):

    '''
    These indices are hardcoded, and taken from the
    sorted keys in feature_spec.
    (NIR - Red) / (NIR + Red)
        2 0_nir_mean
        3 0_red_mean
        8 1_nir_mean
        9 1_red_mean
        14 2_nir_mean
        15 2_red_mean
        20 3_nir_mean
        21 3_red_mean
        26 4_nir_mean
        27 4_red_mean
        32 5_nir_mean
        33 5_red_mean
    '''
    out = []
    for nir_idx, red_idx in NDVI_INDICES:
        # Add a small constant in the denominator to ensure
        # NaNs don't occur because of missing data. Missing
        # data (i.e. Landsat 7 scan line failure) is represented as 0
        # in TFRecord files. Adding \{epsilon} will barely 
        # change the non-missing data, and will make sure missing data
        # is still 0 when it's fed into the model.
        ndvi = (image_stack[:,:, nir_idx] - image_stack[:,:, red_idx]) /\
                (image_stack[:,:, nir_idx] + image_stack[:,:, red_idx] + 1e-8) 
        out.append(ndvi)
    return tf.concat((image_stack, tf.stack(out, axis=-1)), axis=-1)


def filter_list_into_classes(lst):
    out = defaultdict(list)
    for f in lst:
        if 'irrigated' in f and 'unirrigated' not in f:
            out['irrigated'].append(f)
        if 'unirrigated' in f:
            out['unirrigated'].append(f)
        if 'fallow' in f:
            out['fallow'].append(f)
        if 'uncultivated' in f:
            out['uncultivated'].append(f)
        if 'wetlands' in f:
            out['wetlands'].append(f)

    return out

def _assign_weight(name):

    if 'irrigated' in name and 'unirrigated' not in name:
        return 0.3
    if 'unirrigated' in name:
        return 0.7 / 3
    if 'wetlands' in name:
        return 0.7 / 3
    if 'uncultivated' in name:
        return 0.7 / 3
    if 'fallow' in name:
        return 0.165

def make_validation_dataset(root, add_ndvi, batch_size, year,
        n_classes, buffer_size):
    pattern = "*gz"
    training_root = os.path.join(root, pattern)
    files = tf.io.gfile.glob(training_root)

    if year is not None:
        print(len(files))
        files = [f for f in files if year in f]
        print(len(files))

    datasets = get_dataset(files, add_ndvi, n_classes).shuffle(buffer_size).batch(batch_size)
    return datasets

def make_balanced_training_dataset(root,
        add_ndvi,
        batch_size, 
        sample_weights, 
        year,
        buffer_size, 
        n_classes):

    pattern = "*gz"
    datasets = []
    files = tf.io.gfile.glob(os.path.join(root, pattern))

    if year is not None:
        print(len(files))
        files = [f for f in files if year in f]
        print(len(files))

    files = filter_list_into_classes(files) 
    
    weights = []
    for class_name, file_list in files.items():
        dataset = get_dataset(file_list, add_ndvi, n_classes)
        datasets.append(dataset.shuffle(buffer_size).repeat())
        weights.append(_assign_weight(class_name))

    dataset = tf.data.experimental.sample_from_datasets(datasets,
            weights=weights).batch(batch_size)
    return dataset


if __name__ == '__main__':
    pass
