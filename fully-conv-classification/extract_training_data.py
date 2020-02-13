import numpy as np
import numpy.ma as ma
import os
import time
import pickle
import warnings
import argparse
import pdb
import matplotlib.pyplot as plt
import datetime

from glob import glob
from random import sample, shuffle, choice
from scipy.ndimage.morphology import distance_transform_edt
from rasterio import open as rasopen, band
from rasterio.errors import RasterioIOError, CRSError
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage import transform
from sat_image.warped_vrt import warp_single_image
from multiprocessing import Pool 
from collections import defaultdict

from runspec import (landsat_rasters, climate_rasters, mask_rasters, assign_shapefile_class_code,
        assign_shapefile_year, cdl_crop_values, cdl_non_crop_values)
from data_utils import (load_raster, paths_map_multiple_scenes, stack_rasters, create_image_stack,
        stack_rasters_multiprocess, download_from_pr, paths_mapping_single_scene, mean_of_three,
        median_of_three, stack_rasters_single_scene, _parse_landsat_capture_date)
from shapefile_utils import (get_shapefile_path_row, mask_raster_to_shapefile,
        filter_shapefile_overlapping, mask_raster_to_features)


def distance_map(mask):
    mask = mask.copy().astype(bool)
    mask = ~mask # make the non-masked areas masked
    distances = distance_transform_edt(mask) # ask where the closest masked pixel is
    return distances


class DataTile(object):

    def __init__(self, data, one_hot, class_code, cdl_mask):
        self.dict = {}
        print(data.dtype, one_hot.dtype)
        self.dict['data'] = data.astype(np.uint16)
        self.dict['one_hot'] = one_hot.astype(np.uint16)
        self.dict['class_code'] = class_code
        self.dict['cdl'] = cdl_mask

    def to_pickle(self, training_directory):
        if not os.path.isdir(training_directory):
            os.mkdir(training_directory)
        template = os.path.join(training_directory,
                'class_{}_data/'.format(self.dict['class_code']))
        if not os.path.isdir(template):
            os.mkdir(template)
        outfile = os.path.join(template, str(time.time()) + ".pkl")
        if not os.path.isfile(outfile):
            with open(outfile, 'wb') as f:
                pickle.dump(self.dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError()


def _pickle_datatile(datatile, training_directory):
        template = os.path.join(training_directory,
                'class_{}_data/'.format(datatile.dict['class_code']))
        if not os.path.isdir(template):
            os.mkdir(template)
        outfile = os.path.join(template, str(time.time()) + ".pkl")
        if not os.path.isfile(outfile):
            with open(outfile, 'wb') as f:
                pickle.dump(datatile.dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pass


def concatenate_fmasks(image_directory, class_mask, class_mask_geo, nodata=0, target_directory=None):
    ''' ``Fmasks'' are masks of clouds and water. We don't want clouds/water in
    the training set, so this function gets all the fmasks for a landsat
    scene (contained in image_directory), and merges them into one raster. 
    They may not be the same size, so warp_vrt is used to make them align. 
    '''
    class_mask = class_mask.copy()
    paths = []
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            for suffix in mask_rasters():
                if f.endswith(suffix):
                    pth = os.path.join(dirpath, f)
                    paths.append(pth)
    for fmask_file in paths:
        fmask, _ = load_raster(fmask_file)
        # clouds, water present where fmask == 1.
        try:
            class_mask = ma.masked_where(fmask == 1, class_mask)
        except (ValueError, IndexError) as e:
            fmask = warp_single_image(fmask_file, class_mask_geo)
            class_mask = ma.masked_where(fmask == 1, class_mask)

    return class_mask


def create_class_labels(shapefiles, assign_shapefile_class_code, mask_file):
    first = True
    class_labels = None
    for f in shapefiles:
        class_code = assign_shapefile_class_code(f)
        print(f, class_code)
        out, _ = mask_raster_to_shapefile(f, mask_file, return_binary=False)
        if first:
            class_labels = out
            class_labels[~class_labels.mask] = class_code
            first = False
        else:
            class_labels[~out.mask] = class_code
    return class_labels


def _days_from_january_raster(date, target_shape):
    begin = datetime.date(date.year, 1, 1)
    diff = date - begin
    days = diff.days
    out = np.zeros(target_shape)
    out += days
    return out


def concatenate_fmasks_single_scene(class_labels, image_directory, target_date, class_mask_geo):

    date = None
    for d in os.listdir(image_directory):
        if os.path.isdir(os.path.join(image_directory, d)):
            try:
                date = _parse_landsat_capture_date(d)
            except Exception as e:
                print(e)
                continue
            if date == target_date:
                landsat_directory = d
                break
    class_mask = class_labels.copy()
    paths = []
    for dirpath, dirnames, filenames in os.walk(landsat_directory):
        for f in filenames:
            for suffix in mask_rasters():
                if f.endswith(suffix):
                    pth = os.path.join(dirpath, f)
                    paths.append(pth)
    for fmask_file in paths:
        fmask, _ = load_raster(fmask_file)
        # clouds, water present where fmask == 1.
        try:
            class_mask = ma.masked_where(fmask == 1, class_mask)
        except (ValueError, IndexError) as e:
            fmask = warp_single_image(fmask_file, class_mask_geo)
            class_mask = ma.masked_where(fmask == 1, class_mask)

    return class_mask


def extract_training_data_over_path_row_single_scene(test_train_shapefiles, path, row, year,
        image_directory, training_data_root_directory, n_classes, assign_shapefile_class_code,
        preprocessing_func=None, tile_size=608):

    if not isinstance(test_train_shapefiles, dict):
        raise ValueError("expected dict, got {}".format(type(test_train_shapefiles)))
    
    path_row_year = str(path) + '_' + str(row) +  '_' + str(year)
    image_path = os.path.join(image_directory, path_row_year)
    if not os.path.isdir(image_path):
        download_from_pr(path, row, year, image_directory)
    image_path_maps = paths_mapping_single_scene(image_path)
    mask_file = _random_tif_from_directory(image_path)
    mask, mask_meta = load_raster(mask_file)
    mask = np.zeros_like(mask).astype(np.int)
    cdl_path = os.path.join(image_path, 'cdl_mask.tif')
    cdl_raster, cdl_meta = load_raster(cdl_path)
    if mask.shape != cdl_raster.shape:
        cdl_raster = warp_single_image(cdl_path, mask_meta)
    cdl_raster = np.swapaxes(cdl_raster, 0, 2)
    for key, shapefiles in test_train_shapefiles.items():
        try:
            class_labels = create_class_labels(shapefiles, assign_shapefile_class_code, mask_file)
        except TypeError as e:
            print(image_directory)
            download_from_pr(path, row, year, image_directory)
            print(e)
        if key.lower() not in ('test', 'train'):
            raise ValueError("expected key to be one of case-insenstive {test, train},\
            got {}".format(key))
        begin = datetime.date(year=year, month=6, day=15)
        end = datetime.date(year=year, month=9, day=1)
        for date, paths_map in image_path_maps.items():
            if date < begin or date > end:
                print('skipping:', date)
                continue
            try:
                date_raster = _days_from_january_raster(date, target_shape=mask.shape)
                date_raster = np.swapaxes(date_raster, 0, 2)
                image_stack = stack_rasters_single_scene(paths_map, target_geo=mask_meta,
                    target_shape=mask.shape)
                image_stack = np.swapaxes(image_stack, 0, 2)
                image_stack = np.dstack((image_stack, date_raster))
            except RasterioIOError as e:
                print("Redownload images for", path_row_year)
                print(e)
                return
            training_data_directory = os.path.join(training_data_root_directory, key)
            class_labels_single_scene = concatenate_fmasks_single_scene(class_labels,
                    image_path, date, mask_meta) 
            class_labels_single_scene = np.swapaxes(class_labels_single_scene, 0, 2)
            class_labels_single_scene = np.squeeze(class_labels_single_scene)
            tiles_y, tiles_x = _target_indices_from_class_labels(class_labels_single_scene,
                    tile_size)
            _save_training_data_from_indices(image_stack, class_labels_single_scene,
                    cdl_raster, training_data_directory, n_classes, tiles_x, tiles_y,
                    tile_size)



def extract_training_data_over_path_row(test_train_shapefiles, path, row, year, image_directory,
        training_data_root_directory, n_classes, assign_shapefile_class_code, 
        preprocessing_func=None, tile_size=608, use_fmasks=False, use_cdl=False):

    if not isinstance(test_train_shapefiles, dict):
        raise ValueError("expected dict, got {}".format(type(test_train_shapefiles)))
    
    path_row_year = str(path) + '_' + str(row) +  '_' + str(year)
    image_path = os.path.join(image_directory, path_row_year)
    if not os.path.isdir(image_path):
        download_from_pr(path, row, year, image_directory)
    image_path_maps = paths_map_multiple_scenes(image_path)
    mask_file = _random_tif_from_directory(image_path)
    mask, mask_meta = load_raster(mask_file)
    mask = np.zeros_like(mask).astype(np.int)
    if use_cdl:
        cdl_path = os.path.join(image_path, 'cdl_mask.tif')
        cdl_raster, cdl_meta = load_raster(cdl_path)
        if mask.shape != cdl_raster.shape:
            cdl_raster = warp_single_image(cdl_path, mask_meta)
        cdl_raster = np.swapaxes(cdl_raster, 0, 2)
    try:
        image_stack = create_image_stack(image_path_maps)
    except CRSError as e:
        print(e)
        return
    image_stack = np.swapaxes(image_stack, 0, 2)
    for key, shapefiles in test_train_shapefiles.items():
        if key.lower() not in ('test', 'train'):
            raise ValueError("expected key to be one of case-insenstive {test, train},\
            got {}".format(key))
        training_data_directory = os.path.join(training_data_root_directory, key)
        class_labels = create_class_labels(shapefiles, assign_shapefile_class_code)
        if use_fmasks:
            class_labels = concatenate_fmasks(image_path, class_labels, mask_meta) 
        class_labels = np.swapaxes(class_labels, 0, 2)
        class_labels = np.squeeze(class_labels)
        tiles_y, tiles_x = _target_indices_from_class_labels(class_labels, tile_size)
        _save_training_data_from_indices(image_stack, class_labels, 
                training_data_directory, n_classes, tiles_x, tiles_y, tile_size)


def _target_indices_from_class_labels(class_labels, tile_size):
    where = np.nonzero(~class_labels.mask)
    max_y = np.max(where[0])
    min_y = np.min(where[0])
    max_x = np.max(where[1])
    min_x = np.min(where[1])
    max_y += (tile_size - ((max_y - min_y) % tile_size))
    max_x += (tile_size - ((max_x - min_x) % tile_size))
    tiles_y = range(min_y, max_y, tile_size)
    tiles_x = range(min_x, max_x, tile_size)
    return tiles_y, tiles_x


def _assign_class_code_to_tile(class_label_tile):
    if np.any(class_label_tile == 3):
        return 3
    if np.all(class_label_tile != 0):
        unique, unique_count = np.unique(class_label_tile, return_counts=True)
        unique = unique[:-1] # assume np.ma.masked is last.
        unique_count = unique_count[:-1]
        if len(unique_count) == 0:
            return None
        return unique[np.argmax(unique_count)]
    # if a tile has any irrigated pixels, return 0.
    return 0


def _save_training_data_from_indices(image_stack, class_labels, 
        training_data_directory, n_classes, indices_y, indices_x, tile_size):
    out = []
    for i in indices_x:
        for j in indices_y:
            class_label_tile = class_labels[i:i+tile_size, j:j+tile_size]
            shape = class_label_tile.shape
            if np.all(class_label_tile.mask):
                continue
            if (shape[0], shape[1]) != (tile_size, tile_size):
                continue
            class_code = _assign_class_code_to_tile(class_label_tile)
            if class_code is None:
                continue
            sub_one_hot = _one_hot_from_labels(class_label_tile, n_classes)
            sub_image_stack = image_stack[i:i+tile_size, j:j+tile_size, :]
            sub_image_stack = image_stack[i:i+tile_size, j:j+tile_size, :]
            dt = DataTile(sub_image_stack, sub_one_hot, class_code)
            out.append(dt)
            if len(out) > 50:
                with Pool() as pool:
                    td = [training_data_directory]*len(out)
                    pool.starmap(_pickle_datatile, zip(out, td))
                out = []
    if len(out):
        with Pool() as pool:
            td = [training_data_directory]*len(out)
            pool.starmap(_pickle_datatile, zip(out, td))
            out = []


def _random_tif_from_directory(image_directory):

    bleh = os.listdir(image_directory)
    for d in bleh:
        if os.path.isdir(os.path.join(image_directory, d)):
            tiffs = glob(os.path.join(os.path.join(image_directory, d), "*.TIF"))
            tiffs = [tif for tif in tiffs if 'BQA' not in tif]
            break
    shuffle(tiffs)
    return tiffs[0]


def min_data_tiles_to_cover_labels_plot(shapefiles, path, row, year, image_directory, tile_size=608):
    path_row_year = "_".join([str(path), str(row), str(year)])
    image_directory = os.path.join(image_directory, path_row_year)
    mask_file = _random_tif_from_directory(image_directory)
    mask, mask_meta = load_raster(mask_file)
    mask = np.zeros_like(mask).astype(np.int)
    first = True
    class_labels = None
    if not isinstance(shapefiles, list):
        shapefiles = [shapefiles]
    for f in shapefiles:
        class_code = assign_shapefile_class_code(f)
        out, _ = mask_raster_to_shapefile(f, mask_file, return_binary=False)
        if first:
            class_labels = out
            class_labels[~class_labels.mask] = class_code
            first = False
        else:
            class_labels[~out.mask] = class_code
    class_labels = concatenate_fmasks(image_directory, class_labels, mask_meta) 
    where = np.nonzero(~class_labels.mask[0])
    max_y = np.max(where[0])
    min_y = np.min(where[0])
    max_x = np.max(where[1])
    min_x = np.min(where[1])
    frac = np.count_nonzero(~class_labels.mask)/(class_labels.shape[1]*class_labels.shape[2])

    max_y += (tile_size - ((max_y - min_y) % tile_size))
    max_x += (tile_size - ((max_x - min_x) % tile_size))

    tiles_y = range(min_y, max_y, tile_size)
    tiles_x = range(min_x, max_x, tile_size)

    plt.plot([max_x, max_x], [max_y, min_y], 'b', linewidth=2)
    plt.plot([min_x, min_x], [max_y, min_y], 'b', linewidth=2)
    plt.plot([min_x, max_x], [max_y, max_y], 'b', linewidth=2)
    plt.plot([min_x, max_x], [min_y, min_y], 'b', linewidth=2)

    y_min = [min_x] * len(tiles_y)
    y_max = [max_x] * len(tiles_y)
    for t, mn, mx in zip(tiles_y, y_min, y_max):
        plt.plot([mn, mx], [t, t], 'r')

    x_min = [min_y] * len(tiles_x)
    x_max = [max_y] * len(tiles_x)
    for t, mn, mx in zip(tiles_x, x_min, x_max):
        plt.plot([t, t], [mn, mx], 'r')

    plt.imshow(class_labels[0])
    plt.title('path/row: {} {} percent data pixels: {:.3f}'.format(path, row, frac))
    plt.colorbar()
    plt.show()


def _one_hot_from_labels(labels, n_classes):
    one_hot = np.zeros((labels.shape[0], labels.shape[1], n_classes))
    for class_code in range(n_classes):
        one_hot[:, :, class_code][labels == class_code] = 1
        # if class_code == 1: # apply border class to only irrigated pixels
        #     border_labels = make_border_labels(one_hot[:, :, 1], border_width=1)
        #     border_labels = border_labels.astype(np.uint8)
        #     one_hot[:, :, n_classes-1][border_labels == 1] = 1
    return one_hot.astype(np.int)


def _weights_from_one_hot(one_hot, n_classes):
    weights = np.zeros_like(one_hot)
    tmp = np.sum(one_hot, 2)
    for i in range(n_classes):
        weights[:, :, i] = tmp
    return weights.astype(bool)


def _one_hot_from_shapefile(shapefile, mask_file, shapefile_class_code, n_classes):
    class_labels, _ = mask_raster_to_shapefile(shapefile, mask_file, return_binary=False)
    if class_labels.mask.all():
        return None
    one_hot = _one_hot_from_labels(class_labels, shapefile_class_code, n_classes)
    return one_hot


def _check_dimensions_and_min_pixels(sub_one_hot, class_code, tile_size):
    # 200 is the minimum amount of pixels required to save the data.
    if sub_one_hot.shape[0] != tile_size or sub_one_hot.shape[1] != tile_size:
        return False
    xx = np.where(sub_one_hot == class_code)
    if len(xx[0]) == 0:
        return False
    return True


def all_matching_shapefiles(to_match, shapefile_directory, assign_shapefile_year):
    out = []
    pr = get_shapefile_path_row(to_match)
    year = assign_shapefile_year(to_match)
    for f in glob(os.path.join(shapefile_directory, "*.shp")):
        if get_shapefile_path_row(f) == pr and assign_shapefile_year(f) == year:
            out.append(f)
    return out


def make_border_labels(mask, border_width):
    ''' Border width: Pixel width. '''
    dm = distance_map(mask)
    dm[dm > border_width] = 0
    return dm


if __name__ == '__main__':
    
    image_directory = '/home/thomas/ssd/montana_landsat_data_may_to_october'
    shapefiles = glob('shapefile_data/test/*.shp') + glob('shapefile_data/train/*.shp')
    training_root_directory = '/home/thomas/ssd/all_images_no_fmask/'

    if not os.path.isdir(training_root_directory):
        os.makedirs(training_root_directory)

    n_classes = 3
    done = set()
    
    go = False

    for f in shapefiles:
        if f in done:
            continue
        test_shapefiles = all_matching_shapefiles(f, 'shapefile_data/test/', assign_shapefile_year)
        train_shapefiles = all_matching_shapefiles(f, 'shapefile_data/train/', assign_shapefile_year)
        for e in test_shapefiles + train_shapefiles:
            done.add(e)
        bs = os.path.splitext(os.path.basename(f))[0]
        _, path, row = bs[-7:].split("_")
        if not (int(path) == 37 and int(row) == 29):
            continue
        year = assign_shapefile_year(f)
        print("extracting data for", path, row, year)
        test_train_shapefiles = {'test':test_shapefiles, 'train':train_shapefiles}
        extract_training_data_over_path_row(test_train_shapefiles, path, row, 
                year, image_directory, training_root_directory, n_classes,
                assign_shapefile_class_code)
