import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import gc
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
import numpy as np
from pdb import set_trace
from glob import glob
from sys import stdout, exit
from rasterio import open as rasopen
from tensorflow.keras.models import load_model
from models import unet
from random import shuffle
from extract_training_data import stack_images_from_list_of_filenames_sorted_by_date, bin_images_into_path_row_year, parse_date


def iterate_over_image_and_evaluate_patchwise_cnn(image_stack, model_path, out_filename, out_meta,
        n_classes, tile_size=24):

    model= unet((None, None, 98), n_classes=3, initial_exp=5)
    model.load_weights(model_path)
    
    print(image_stack.shape)
    image_stack = np.swapaxes(image_stack, 0, 2)
    image_stack = np.expand_dims(image_stack, 0)
    print(image_stack.shape)

    predictions = np.zeros((image_stack.shape[1], image_stack.shape[2], n_classes))
    for i in range(0, image_stack.shape[1]-tile_size, tile_size):
        for j in range(0, image_stack.shape[2]-tile_size, tile_size):
            image_tile = image_stack[:, i:i+tile_size, j:j+tile_size, :]
            if np.all(image_tile == 0):
                continue
            predictions[i:i+tile_size, j:j+tile_size, :] = np.squeeze(model.predict(image_tile))
        stdout.write("{:.3f}\r".format(i/image_stack.shape[1]))


    predictions = np.swapaxes(predictions, 0, 2)
    out_meta.update({'count':n_classes, 'dtype':np.float64})
    with rasopen(out_filename, "w", **out_meta) as dst:
        dst.write(predictions)


def iterate_over_image_and_evaluate_patchwise_lstm_cnn(image_stack, model_path, out_filename, out_meta,
        n_classes, tile_size=24):

    model = load_model(model_path, custom_objects={'m_acc':m_acc})

    timeseries = []
    for i in range(0, image_stack.shape[0]-3, 3):
        timeseries.append(image_stack[i:i+3])
    
    timeseries = np.asarray(timeseries)
    timeseries = np.swapaxes(timeseries, 1, 3)
    timeseries = np.expand_dims(timeseries, 0)
    print(timeseries.shape)

    for start_idx in range(0, timeseries.shape[1]-12):
        predictions = np.zeros((timeseries.shape[2], timeseries.shape[3], n_classes))
        timeseries_copy = timeseries[:, start_idx:start_idx+12, :, :, :]
        for i in range(0, timeseries_copy.shape[2]-tile_size, tile_size):
            for j in range(0, timeseries_copy.shape[3]-tile_size, tile_size):
                image_tile = timeseries_copy[:, :, i:i+tile_size, j:j+tile_size, :]
                if np.all(image_tile == 0):
                    continue
                preds = np.squeeze(model.predict(image_tile))
                predictions[i:i+tile_size, j:j+tile_size, :] = np.sum(preds, axis=0)
            stdout.write("{}, {:.3f}\r".format(start_idx, i/timeseries_copy.shape[2]))


        predictions = np.swapaxes(predictions, 0, 2)
        out_meta.update({'count':n_classes, 'dtype':np.float64})
        with rasopen(out_filename, "w", **out_meta) as dst:
            dst.write(predictions)


if __name__ == '__main__':

    image_directory = '/home/thomas/ssd/stacked_images_2015_mt/'
    images = glob(os.path.join(image_directory, "*tif"))
    path_row_to_images = bin_images_into_path_row_year(images)
    start_idx = 7
    year = 2013

    base = './current_models/non-recurrent/full-unet-random_start_date-diff-lr-with-centroids/'
    model_name =  'model_0.969-0.910.h5'   
    model_path = base + model_name

    for target_path_row_year in path_row_to_images:

        image_filenames = path_row_to_images[target_path_row_year]
        shuffle(image_filenames)
        image_filenames = np.random.choice(image_filenames, size=14, replace=False)
        out_filename = './7start/{}-{}.tif'.format(target_path_row_year, model_name)
        if os.path.isfile(out_filename):
            continue
        if start_idx is not None:
            image_filenames = image_filenames[start_idx:start_idx + 14]
        if len(image_filenames) < 14:
            print('no images for {}, exiting'.format(target_path_row_year))
            continue
        image_filenames = sorted(image_filenames, key=lambda x: parse_date(x))
        image_stack, target_meta, target_fname, oo = stack_images_from_list_of_filenames_sorted_by_date(image_filenames)

        if target_meta['crs'] is None:
            target_meta = oo

        n_classes = 3
        lstm = False
        if lstm:
            iterate_over_image_and_evaluate_patchwise_lstm_cnn(image_stack, model_path,
                    out_filename, target_meta, 3, tile_size=224)
        else:
            iterate_over_image_and_evaluate_patchwise_cnn(image_stack, model_path,
                     out_filename, target_meta, 3, tile_size=224)
        gc.collect()
