import os
import tensorflow as tf
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from pdb import set_trace
from glob import glob
from sys import stdout, exit
from tensorflow.keras.models import load_model
from argparse import ArgumentParser

import utils.feature_spec as feature_spec
from models.unet import unet

NDVI_INDICES = [(2, 3), (8, 9), (14, 15), (20, 21), (26, 27), (32, 33)]


def iterate_over_image_and_evaluate_patchwise(image_stack, model, out_filename, out_meta,
        n_classes, tile_size, chunk_size):

    # pad on both sides by tile_size // 2
    pad_width  = tile_size 
    pad_height = tile_size 
    image_stack = np.pad(image_stack, 
            ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

    image_stack = np.expand_dims(np.transpose(image_stack, [1, 2, 0]), 0)
    chunk_size = chunk_size
    diff = (tile_size - chunk_size) // 2
    predictions = np.zeros((image_stack.shape[1], image_stack.shape[2], n_classes))
    # I need to do a seamless prediction.
    for i in range(tile_size//2, image_stack.shape[1]-tile_size//2, chunk_size):
        for j in range(tile_size//2, image_stack.shape[2]-tile_size//2, chunk_size):
            image_tile = tf.convert_to_tensor(image_stack[:, i-tile_size//2:i+tile_size//2, j-tile_size//2:j+tile_size//2, :])
            if np.all(image_tile == 0):
                continue
            preds = np.squeeze(model(image_tile)['softmax'])
            predictions[i-chunk_size//2:i+chunk_size//2, j-chunk_size//2:j+chunk_size//2, :] += preds[diff:-diff, diff:-diff, :]
        stdout.write("{:.3f}\r".format(i/image_stack.shape[1]))


    predictions = predictions.transpose((2, 0, 1))
    predictions = predictions[:, pad_width:-pad_height, pad_width:-pad_width] 
    predictions = np.round(predictions*255).astype(np.uint8)
    out_meta.update({'count':n_classes, 'dtype':np.uint8})
    with rasterio.open(out_filename, "w", **out_meta) as dst:
        dst.write(predictions)

def prepare_raster(f, ndvi):

    with rasterio.open(f, 'r') as src:
        image_stack = src.read()
        target_meta = src.meta
        descriptions = src.descriptions
        # print(src.descriptions)
        indices = np.argsort(descriptions)

    features = set(feature_spec.features())

    descriptions = np.asarray(descriptions)[indices]

    final_indices = []

    for d, idx in zip(descriptions, indices):
        if d in features:
            final_indices.append(idx)

    image_stack = image_stack.astype(np.float32)
    image_stack = image_stack[np.asarray(final_indices)] * 0.0001
    image_stack[np.isnan(image_stack)] = 0
    if ndvi:
        out = []
        for nir_idx, red_idx in NDVI_INDICES:
            # Add a small constant in the denominator to ensure
            # NaNs don't occur because of missing data. Missing
            # data (i.e. Landsat 7 scan line failure) is represented as 0
            # in TFRecord files. Adding \{epsilon} will barely 
            # change the non-missing data, and will make sure missing data
            # is still 0 when it's fed into the model.
            ndvi = (image_stack[nir_idx] - image_stack[red_idx]) /\
                    (image_stack[nir_idx] + image_stack[red_idx] + 1e-6) 
            out.append(ndvi)
        # print(image_stack.shape, np.stack(out).shape)
        image_stack = np.concatenate((image_stack, np.stack(out)), axis=0)

    return image_stack, target_meta


def model_predictions(model_path, 
                      data_directory, 
                      image_file,
                      year,
                      out_directory,
                      n_classes,
                      use_cuda,
                      tile_size,
                      chunk_size,
                      show_logs,
                      ndvi):

    if not show_logs:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if data_directory is not None:
        files = glob(os.path.join(data_directory, "*tif"))
        if year is not None:
            files = [f for f in files if str(year) in f]
    elif image_file is not None:
        files = [image_file]
    else:
        raise ValueError('Either data directory or image file must be provided')

    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures["serving_default"]
    n_classes = n_classes
    
    out_directory = out_directory

    if not os.path.isdir(out_directory):
        os.makedirs(out_directory, exist_ok=True)

    for f in files:
        try:
            image_stack, target_meta = prepare_raster(f, ndvi)
        except (rasterio.errors.RasterioIOError, AttributeError, TypeError) as e:
            print(f)
            print(e)
            continue
        out_filename = 'irr{}'.format(os.path.splitext(os.path.basename(f))[0])
        out_filename +=  os.path.basename(model_path) + ".tif"
        out_filename = os.path.join(out_directory, out_filename)
        if not os.path.isfile(out_filename):
            print(out_filename)
            iterate_over_image_and_evaluate_patchwise(image_stack,
                    model,
                    out_filename,
                    target_meta,
                    n_classes=n_classes,
                    tile_size=tile_size,
                    chunk_size=chunk_size)

        else:
            print('file', f, 'already predicted, residing in', out_filename)


if __name__ == '__main__':

    ap = ArgumentParser()

    ap.add_argument('--model-path', required=True)
    ap.add_argument('--data-directory', type=str)
    ap.add_argument('--image-file', type=str)
    ap.add_argument('--year', type=str)
    ap.add_argument('--out-directory', required=True)
    ap.add_argument('--n-classes', type=int, required=True)
    ap.add_argument('--use-cuda', action='store_true')
    ap.add_argument('--tile-size', type=int, default=608)
    ap.add_argument('--chunk-size', type=int, default=256)
    ap.add_argument('--show-logs', action='store_true')
    ap.add_argument('--ndvi', action='store_true')

    args = ap.parse_args()
    model_predictions(args.model_path,
                      args.data_directory, 
                      args.image_file,
                      args.year,
                      args.out_directory,
                      args.n_classes,
                      args.use_cuda,
                      args.tile_size,
                      args.chunk_size,
                      args.show_logs,
                      args.ndvi)
