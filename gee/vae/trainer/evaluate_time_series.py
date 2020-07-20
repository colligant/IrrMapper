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

def iterate_over_image_and_evaluate_patchwise_cnn(image_stack, model_path, out_filename, out_meta,
        n_classes, tile_size=24):

    model= unet((None, None, 36), n_classes=n_classes, initial_exp=4)
    model.load_weights(model_path)
    
    image_stack = tf.transpose(image_stack, [1, 2, 0])
    image_stack = np.expand_dims(image_stack, 0)

    predictions = np.zeros((image_stack.shape[1], image_stack.shape[2], n_classes))
    for i in range(0, image_stack.shape[1]-tile_size, tile_size):
        for j in range(0, image_stack.shape[2]-tile_size, tile_size):
            image_tile = image_stack[:, i:i+tile_size, j:j+tile_size, :]
            if np.all(image_tile == 0):
                continue
            predictions[i:i+tile_size, j:j+tile_size, :] = np.squeeze(model.predict(image_tile))
        stdout.write("{:.3f}\r".format(i/image_stack.shape[1]))


    predictions = predictions.transpose((2, 0, 1))
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


    base = './models/'
    model_name =  'model-0.980-0.919.h5'
    model_path = base + model_name

    files= glob("./image_data/tifs/*tif")
    n_classes = 3
    for f in files:
        if "TF" in f:
            out_filename = 'testamax{}.tif'.format(os.path.splitext(os.path.basename(f))[0])
            out_filename = os.path.join('./image_data/evaluated/', out_filename)
            if not os.path.isfile(out_filename):
                with rasopen(f, 'r') as src:
                    image_stack = src.read()
                    target_meta = src.meta
                    descriptions = src.descriptions
                    indices = np.argsort(descriptions)
                image_stack = image_stack[indices]
                iterate_over_image_and_evaluate_patchwise_cnn(image_stack, model_path,
                         out_filename, target_meta, 3, tile_size=256)
