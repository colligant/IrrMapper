import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
import numpy as np
from pdb import set_trace
from glob import glob
from sys import stdout, exit
from rasterio import open as rasopen
from tensorflow.keras.models import load_model
from extract_training_data import stack_rgb_images_from_list_of_filenames_sorted_by_date, bin_images_into_path_row_year, parse_date
from tifs_and_pngs.losses import m_acc


def iterate_over_image_and_evaluate_patchwise_cnn(image_stack, model_path, out_filename, out_meta,
        n_classes, tile_size=24):

    model = load_model(model_path, custom_objects={'m_acc':m_acc})

    
    print(image_stack.shape)
    image_stack = np.swapaxes(image_stack, 0, 2)
    image_stack = np.expand_dims(image_stack, 0)

    for start_idx in range(0, image_stack.shape[-1]-39, 3):
        predictions = np.zeros((image_stack.shape[1], image_stack.shape[2], n_classes))
        image_stack_copy = image_stack[:, :, :, start_idx:start_idx+39]
        for i in range(0, image_stack_copy.shape[1]-tile_size, tile_size):
            for j in range(0, image_stack_copy.shape[2]-tile_size, tile_size):
                image_tile = image_stack_copy[:, i:i+tile_size, j:j+tile_size, :]
                if np.all(image_tile == 0):
                    continue
                try:
                    predictions[i:i+tile_size, j:j+tile_size, :] = np.squeeze(model.predict(image_tile))
                except Exception as e:
                    set_trace()
            stdout.write("{}, {:.3f}\r".format(start_idx, i/image_stack_copy.shape[1]))


        predictions = np.swapaxes(predictions, 0, 2)
        out_meta.update({'count':n_classes, 'dtype':np.float64})
        out_filename_temp = os.path.splitext(os.path.basename(out_filename))[0] + str(start_idx) + ".tif"
        with rasopen(out_filename_temp, "w", **out_meta) as dst:
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
        out_filename_temp = os.path.splitext(os.path.basename(out_filename))[0] + str(start_idx) + ".tif"
        out_meta.update({'count':n_classes, 'dtype':np.float64})
        with rasopen(out_filename_temp, "w", **out_meta) as dst:
            dst.write(predictions)


if __name__ == '__main__':

    image_directory = '/home/thomas/ssd/rgb-surf/'
    image_directory = '/home/thomas/ssd/rgb-mt-test/'
    images = glob(os.path.join(image_directory, "*tif"))
    path_row_to_images = bin_images_into_path_row_year(images)
    model_name = 'unet-0.939'
    year = 2017
    for year in [2015, 2016, 2017, 2018, 2019]:
        model_path = './tifs_and_pngs/models/non-recurrent/{model_name}.h5'.format(model_name=model_name)
        target_path_row_year = '{}_41_28'.format(year)
        image_filenames = path_row_to_images[target_path_row_year]
        if not len(image_filenames):
            print('no images for {}, exiting'.format(target_path_row_year))
            exit()

        image_filenames = sorted(image_filenames, key=lambda x: parse_date(x))
        image_stack, target_meta, target_fname, _ = stack_rgb_images_from_list_of_filenames_sorted_by_date(image_filenames)

        n_classes = 3
        lstm = False
        out_filename = './imagestack-evaluated/{}-{}.tif'.format(target_path_row_year, model_name)
        if lstm:
            iterate_over_image_and_evaluate_patchwise_lstm_cnn(image_stack, model_path,
                    out_filename, target_meta, 3, tile_size=216)
        else:
            iterate_over_image_and_evaluate_patchwise_cnn(image_stack, model_path,
                     out_filename, target_meta, 3, tile_size=216)
