import os
import sys
import numpy as np
import rasterio
import datetime
import matplotlib.pyplot as plt
import numpy.ma as ma

from collections import defaultdict
from copy import deepcopy
from random import shuffle
from glob import glob
from data_utils import _parse_landsat_capture_date as parse_landsat_capture_date
from sat_image.warped_vrt import warp_single_image

from shapefile_utils import mask_raster_to_shapefile
from extract_training_data import create_class_labels

targets = ('B4.TIF', 'B3.TIF', 'B2.TIF')


def save_rgb_image(rgb_tif_dict, outfile):

    if os.path.isfile(outfile):
#        print('already created', outfile)
        return
    if len(rgb_tif_dict) == 0:
        return
    print('creating', outfile)
    first = True
    for i, band in enumerate(targets):
        try:
            filename = rgb_tif_dict[band]
        except KeyError as e:
            print(e)
            return
        with rasterio.open(filename, 'r') as src:
            arr = src.read()
            meta = src.meta.copy()
        if first:
            rgb_array = np.zeros((3, arr.shape[1], arr.shape[2]), dtype=meta['dtype'])
            rgb_array[i] = arr
            first = False
        else:
            rgb_array[i] = arr
        meta.update({'count':3})
        with rasterio.open(outfile, 'w', **meta) as dst:
            dst.write(rgb_array)


landsat_path = '/home/thomas/share/landsat_test/'
out_path = '/home/thomas/ssd/rgb-landsat-entire-year/'
in_path = '/home/thomas/ssd/rgb-landsat-entire-year/'
# save_all_rgb(landsat_path, out_path)
test_or_train = 'train'
target_path_row = '34_28'
path_row_to_tif = defaultdict(list)

for tif in sorted(glob(in_path + "*TIF")):
    path_row = os.path.basename(tif).split('_')
    path, row = path_row[1:3]
    pr = path + "_" + row
    date = filename_to_date(tif)
    path_row_to_tif[pr].append(tif)

for val in path_row_to_tif.values():
    val = sorted(val, key=filename_to_date)

test_case = path_row_to_tif[target_path_row]


image_stack, og_meta, og_fname, meta = stack_rgb_images_from_list_of_filenames_sorted_by_date(test_case)

iis = np.arange(0, image_stack.shape[1], 608)
jjs = np.arange(0, image_stack.shape[2], 608)
shuffle(iis)
shuffle(jjs)

#for ii in iis:
#    for jj in jjs:
#        for idx, i in enumerate(range(0, image_stack.shape[0]-3, 3)):
#            i1 = image_stack[i:i+3, ii:ii+608, jj:jj+608].T
#            i2 = image_stack[i+3:i+6, ii:ii+608, jj:jj+608].T
#            if np.any(i1 == 0) or np.any(i2 == 0):
#                continue
#            fig, ax = plt.subplots(ncols=2)
#            ax[0].imshow(i1/np.max(i1))
#            ax[1].imshow(i2/np.max(i2))
#            ax[0].set_title(i)
#            ax[1].set_title(test_case[idx])
#            plt.show()

def cc(shape):
    if 'irrigated' in shape and 'unirrigated' not in shape:
        return 0
    if 'unirrigated' in shape or 'fallow' in shape:
        return 1
    if 'uncultivated' in shape:
        return 2
    return 2

shapefiles = glob('shapefile_data/{}/*shp'.format(test_or_train))
shapefiles = [s for s in shapefiles if target_path_row in s]
class_labels = create_class_labels(shapefiles, cc, og_fname)
class_labels = np.sum(class_labels, axis=0) / class_labels.shape[0]
class_labels = class_labels.astype(np.uint8)


tile_size = 108 # 54 27 
out_data_dir = 'segmentation'
out_mask = 'tifs_and_pngs/data/{}/{}/masks/'.format(out_data_dir, test_or_train)
out_image = 'tifs_and_pngs/data/{}/{}/images/'.format(out_data_dir, test_or_train)
meta.update({'count': image_stack.shape[0], 'width':tile_size, 'height':tile_size})
label_meta = deepcopy(meta)
label_meta.update({'count':1, 'dtype':np.uint8})

class_code_to_class_name = {0:'irrigated', 1:'unirrigated', 2:'uncultivated', 3:'unknown'}

for i in range(0, class_labels.shape[0]-tile_size, tile_size):
    for j in range(0, class_labels.shape[1]-tile_size, tile_size):
        sub_label = class_labels[i:i+tile_size, j:j+tile_size]
        out_tif = image_stack[:, i:i+tile_size, j:j+tile_size]
        if np.any(out_tif == 0):
            continue
        if np.all(sub_label.mask):
            continue
        if np.sum(~sub_label.mask) / np.sum(sub_label.mask) < 0.9:
            continue
        values, counts = np.unique(sub_label, return_counts=True)
        idx = np.argsort(counts)
        values = values[idx]
        values = values[~values.mask]
        try:
            class_idx = values[-1]
        except IndexError:
            class_idx = 3
        try:
            class_name = class_code_to_class_name[class_idx]
        except KeyError:
            class_name = 'unknown'
        sub_label[sub_label.mask] = 255 # max uint8
        if np.any(sub_label == 0):
            class_name = 'irrigated'
        out_image_dir = os.path.join(out_image, '{}'.format(class_name))
        if len(values) == 3:
            print(out_image_dir + 'test_{}_{}.tif'.format(i, j))
        if not os.path.isdir(out_image_dir):
            os.makedirs(out_image_dir)
        out_mask_dir = os.path.join(out_mask, '{}'.format(class_name))
        if not os.path.isdir(out_mask_dir):
            os.makedirs(out_mask_dir)
        with rasterio.open(os.path.join(out_image_dir, 'test_{}_{}.tif'.format(i, j)),
                'w', **meta) as dst:
            dst.write(out_tif)
        with rasterio.open(os.path.join(out_mask_dir, 'test_{}_{}.tif'.format(i, j)),
                'w', **label_meta) as dst:
            dst.write(np.expand_dims(sub_label, axis=0))
