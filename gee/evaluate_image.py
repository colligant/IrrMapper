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


def iterate_over_image_and_evaluate_patchwise(image_stack, model_path, out_filename, out_meta,
        n_classes, tile_size, chunk_size):

    # pad on both sides by tile_size // 2
    pad_width  = tile_size 
    pad_height = tile_size 
    image_stack = np.pad(image_stack, 
            ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

    image_stack = np.expand_dims(np.transpose(image_stack, [1, 2, 0]), 0)
    print(image_stack.shape)
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

    args = ap.parse_args()

    if not args.show_logs:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.data_directory is not None:
        files = glob(os.path.join(args.data_directory, "*tif"))
        if args.year is not None:
            files = [f for f in files if str(args.year) in f]
    elif args.image_file is not None:
        files = [args.image_file]
    else:
        raise ValueError('Either data directory or image file must be provided')

    loaded = tf.saved_model.load(args.model_path)
    model = loaded.signatures["serving_default"]
    n_classes = args.n_classes
    
    out_directory = args.out_directory

    if not os.path.isdir(out_directory):
        os.makedirs(out_directory, exist_ok=True)


    for f in files:

        out_filename = 'irr{}.tif'.format(os.path.splitext(os.path.basename(f))[0])
        out_filename = os.path.join(out_directory, out_filename)

        if not os.path.isfile(out_filename):
            print(out_filename)
            try:
                with rasterio.open(f, 'r') as src:
                    image_stack = src.read()
                    target_meta = src.meta
                    descriptions = src.descriptions
                    # print(src.descriptions)
                    indices = np.argsort(descriptions)
            except (rasterio.errors.RasterioIOError, AttributeError, TypeError) as e:
                print(f)
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
                    tile_size=args.tile_size,
                    chunk_size=args.chunk_size)

        else:
            print('file', f, 'already predicted, residing in', out_filename)
