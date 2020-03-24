import os
import rasterio as rio
import pdb
import numpy as np
from glob import glob
from copy import deepcopy
from itertools import product
from rasterio import windows

from extract_training_data import stack_rgb_images_from_list_of_filenames_sorted_by_date, bin_images_into_path_row_year, parse_date


image_directory = '/home/thomas/share/rgb-surface-reflectance/'
#shapefiles = glob('shapefile_data/test/*.shp') + glob('shapefile_data/train/*.shp')
#training_root_directory = '/home/thomas/ssd/training-data/'
images = glob(os.path.join(image_directory, "*tif"))
path_row_to_images = bin_images_into_path_row_year(images)

for k,v in path_row_to_images.items():
    print(k)




