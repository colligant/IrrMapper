import tensorflow as tf
import os
import numpy as np
import json
from glob import glob

import feature_spec
from models import unet

features_dict = feature_spec.features_dict()
bands = feature_spec.bands()
features = feature_spec.features()

bands = {}
for k, v in features_dict.items():
    if k != 'constant':
        bands[k] = v


def parse_tfrecord(example_proto):
  """the parsing function.
  read a serialized example into the structure defined by features_dict.
  args:
    example_proto: a serialized example.
  returns:
    a dictionary of tensors, keyed by feature name.
  """
  return tf.io.parse_single_example(example_proto, bands)

def to_tuple(inputs):
  inputsList = [inputs.get(key) for key in sorted(bands)]
  stacked = tf.stack(inputsList, axis=0)
  stacked = tf.transpose(stacked, [1, 2, 0])
  inputs = stacked[:,:,:len(bands)] 
  return (inputs,)

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

def make_dataset(root, batch_size=16, training=True):
    pattern = "*gz"
    datasets = []
    training_root = os.path.join(root, pattern)
    return get_dataset(training_root).batch(1)
files = glob('./tfrecord/*gz') + glob('./tfrecord/*json')
image_files_list = []
json_file = None
for f in files:
  if f.endswith('.tfrecord.gz'):
    image_files_list.append(f)
  elif f.endswith('.json'):
    json_file = f

# Make sure the files are in the right order.
image_files_list.sort()

json_file = './tfrecord/37_28_2015TF-mixer.json'
with open(json_file, 'r') as f:
    st = f.read()
    mixer = json.loads(st)
print(mixer)

writer = tf.io.TFRecordWriter('test.TFRecord')
patches = mixer['totalPatches']
print(patches)
dataset = make_dataset('./tfrecord', 1)
model = unet((None, None, 36), initial_exp=4, n_classes=3)
model.load_weights('./models/model-0.980-0.919.h5')
predictions = model.predict(dataset, steps=1)

# Every patch-worth of predictions we'll dump an example into the output
# file with a single feature that holds our predictions. Since our predictions
# are already in the order of the exported data, the patches we create here
# will also be in the right order.
patch_width = mixer['patchDimensions'][0]
patch_height = mixer['patchDimensions'][1]
patches = mixer['totalPatches']
patch_dimensions_flat = [patch_width * patch_height, 1]
patch = [[], [], [], []]
cur_patch = 1
for prediction in predictions:
  patch[0].append(tf.argmax(prediction, -1))
  patch[1].append(prediction[0][0])
  # Once we've seen a patches-worth of class_ids...
  print('Done with patch ' + str(cur_patch) + ' of ' + str(patches) + '...')
  # Create an example
  example = tf.train.Example(
    features=tf.train.Features(
      feature={
        'prediction': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(prediction)]))
      }
    )
  )
  # Write the example to the file and clear our patch array so it's ready for
  # another batch of class ids
  writer.write(example.SerializeToString())
  patch = [[], [], [], []]
  cur_patch += 1

writer.close()
