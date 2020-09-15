import os
import tensorflow as tf
import numpy as np
import rasterio

from pdb import set_trace
from glob import glob
from sys import stdout, exit
from tensorflow.keras.models import load_model
from argparse import ArgumentParser

import utils.feature_spec as feature_spec
from models.models import unet


def iterate_over_image_and_evaluate_patchwise(image_stack, model_path, out_filename, out_meta,
        n_classes, tile_size=24):

    image_stack = tf.transpose(image_stack, [1, 2, 0])
    image_stack = tf.convert_to_tensor(np.expand_dims(image_stack, 0))

    predictions = np.zeros((image_stack.shape[1], image_stack.shape[2], n_classes))
    for ofs in range(1):
        for i in range(ofs, image_stack.shape[1]-tile_size, tile_size):
            for j in range(ofs, image_stack.shape[2]-tile_size, tile_size):
                image_tile = image_stack[:, i:i+tile_size, j:j+tile_size, :]
                if np.all(image_tile == 0):
                    continue
                preds = np.squeeze(model(image_tile)['softmax'])
                predictions[i:i+tile_size, j:j+tile_size, :] += preds
            stdout.write("{:.3f}\r".format(i/image_stack.shape[1]))


    predictions = predictions.transpose((2, 0, 1))
    out_meta.update({'count':n_classes, 'dtype':np.float64})
    with rasterio.open(out_filename, "w", **out_meta) as dst:
        dst.write(predictions)


if __name__ == '__main__':

    ap = ArgumentParser()

    ap.add_argument('--model-path', required=True)
    ap.add_argument('--data-directory', required=True)
    ap.add_argument('--out-directory', required=True)
    ap.add_argument('--n-classes', type=int, required=True)
    ap.add_argument('--use-cuda', action='store_true')
    ap.add_argument('--show-logs', action='store_true')

    args = ap.parse_args()

    if not args.show_logs:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


    model_path = args.model_path
    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures["serving_default"]
    n_classes = args.n_classes
    
    out_directory = args.out_directory

    if not os.path.isdir(out_directory):
        os.makedirs(out_directory, exist_ok=True)

    files = glob(os.path.join(args.data_directory, "*tif"))

    for f in files:

        out_filename = 'test{}.tif'.format(os.path.splitext(os.path.basename(f))[0])
        out_filename = os.path.join(out_directory, out_filename)

        if not os.path.isfile(out_filename):
            print('Saving image to:', out_filename)
            try:
                with rasterio.open(f, 'r') as src:
                    image_stack = src.read()
                    target_meta = src.meta
                    descriptions = src.descriptions
                    indices = np.argsort(descriptions)
            except rasterio.errors.RasterioIOError as e:
                print(e)
                continue
            features = set(feature_spec.features())

            descriptions = np.asarray(descriptions)[indices]

            final_indices = []

            for d, idx in zip(descriptions, indices):
                if d in features:
                    final_indices.append(idx)

            image_stack = image_stack.astype(np.float32)
            image_stack = image_stack[np.asarray(final_indices)] * 0.0001
            image_stack[np.isnan(image_stack)] = 0
            iterate_over_image_and_evaluate_patchwise(image_stack,
                    model,
                    out_filename,
                    target_meta,
                    n_classes=n_classes,
                    tile_size=608)
