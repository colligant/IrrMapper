import os
import numpy as np
import rasterio
import json
import geopandas as gpd
import pandas as pd

from copy import deepcopy
from glob import glob
from subprocess import check_output
from collections import defaultdict

def merge_rasters_gdal(raster_files, out_filename):
    if not os.path.isfile(out_filename):
        cmd = check_output(['gdal_merge.py', '-of', 'GTiff', '-ot', 'Byte',
            '-o', out_filename] + raster_files)
        print(cmd, len(raster_files), year)


def clip_raster_to_shapefiles_gdal(raster_file, shapefiles, out_directory):

    for f in shapefiles:
        out_filename = os.path.join(out_directory, 
                os.path.splitext(os.path.basename(f))[0] + "_" + year + ".tif")
        if not os.path.isfile(out_filename):
            print('clipping', merged_raster, 'to', f, 'saving to', out_filename)
            cmd = check_output(['gdalwarp', '-of', 'GTiff', '-cutline', f, 
                               '-crop_to_cutline', raster_file, out_filename])

def calc_irr_area(arr):
    amax = np.argmax(arr, axis=0)
    mask = (np.sum(arr, axis=0) == 0)
    amax = amax[~mask]
    irrigated = amax[amax == 0]
    return irrigated.shape[0]


def convert_to_uint16(files):

    for f in files:
        print('converting', f)
        with rasterio.open(f, 'r') as src:
            image_stack = src.read()
            target_meta = deepcopy(src.meta)
            descriptions = src.descriptions
        if image_stack.dtype == np.float16:
            print("didn't need to convert", f)
            continue
        image_stack = image_stack.astype(rasterio.uint16)
        target_meta.update({'dtype':rasterio.uint16})
        with rasterio.open(f, "w", **target_meta) as dst:
            dst.descriptions = descriptions
            dst.write(image_stack)

def plotter():

    import matplotlib.pyplot as plt
    df = pd.read_csv("./montana_irrigated_acreage.csv")
    nass = pd.read_csv("./nass_data.csv")
    years = [2002, 2007, 2012, 2017]
    preds = np.round(df.loc[df.iloc[:, 0].isin(years), :])
    fig, ax = plt.subplots(nrows=6, ncols=10) 
    counties = list(preds.keys())
    counties.remove('year')
    nass_counties = nass.columns
    nass_counties = [s.replace("MT_", '').lower().replace(" ", "_") for s in nass_counties]
    nass.columns = nass_counties
    for i, county in enumerate(counties):
        n = nass[county]
        p = preds[county]
        if i == 55:
            ax[i // 10, i % 10].plot(range(4), n, label='nass')
            ax[i // 10, i % 10].plot(range(4), p, label='preds')
            ax[i // 10, i % 10].axis('off')
            ax[i // 10, i % 10].set_title(county)
            ax[i // 10, i % 10].legend()

        else:
            ax[i // 10, i % 10].plot(range(4), n)
            ax[i // 10, i % 10].plot(range(4), p)
            ax[i // 10, i % 10].set_title(county)
            ax[i // 10, i % 10].axis('off')

plt.show()

if __name__ == '__main__':

    root = "/home/thomas/montana_data/montana-irr-rasters/rasters_by_county/*tif"
    county_to_year_and_area = defaultdict(dict)
    for i, f in enumerate(glob(root)):
        path = os.path.splitext(os.path.basename(f))[0]
        if path.count("_") > 1:
            county_name = path[:path.rfind('_')]
            year = path[path.rfind('_')+1:]
        else:
            county_name = path[:path.find('_')]
            year = path[path.find('_')+1:]
        print(i, county_name, year)
        with rasterio.open(f, 'r') as src:
            arr = src.read()
        county_to_year_and_area[county_name][year] = calc_irr_area(arr)

    with open("county_to_year_and_area.json","w") as f:
        json.dump(county_to_year_and_area, f)

    df = pd.from_dict(county_to_year_and_area)
    df.to_csv('mt_irrsept20.csv')
