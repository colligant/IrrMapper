import os
import numpy as np
import rasterio
import json
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import box, Polygon
from rasterio import merge
from copy import deepcopy
from glob import glob
from subprocess import check_output
from collections import defaultdict

def merge_rasters_gdal(raster_files, out_filename):
    if not len(raster_files):
        return
    if not os.path.isfile(out_filename):
        cmd = check_output(['gdal_merge.py', '-of', 'GTiff', '-ot', 'Byte',
            '-o', out_filename] + raster_files)
        print(cmd)


def merge_split_rasters_copy_band_descriptions(rasters, out_filename):

    if not os.path.isfile(out_filename):
        dsets = []
        for raster in rasters:
            dsets.append(rasterio.open(raster, 'r'))
        merged, out_transform = merge.merge(dsets)

        with rasterio.open(rasters[0], 'r') as src:
            meta = src.meta.copy()
            descriptions = src.descriptions
        meta.update({
            'height': merged.shape[1],
            'width': merged.shape[2],
            'transform': out_transform
            })
        with rasterio.open(out_filename, 'w', **meta) as dst:
            dst.descriptions = descriptions
            dst.write(merged)
        for raster in rasters:
            print('removing', raster)
            os.remove(raster)


def _assign_year(raster, years):
    for year in years:
        if year in raster:
            return year
    raise ValueError('raster {} didn\'t have a year attr.') 



def rename_rasters(rasters, county_shapefile, out_directory):
    osb = os.path.basename
    years = [str(r) for r in range(2000, 2020)]
    gdf = gpd.read_file(county_shapefile)
    gdf = gdf.to_crs('EPSG:5070')
    for raster in rasters:
        year = _assign_year(osb(raster), years)
        with rasterio.open(raster, 'r') as src:
            bounds = src.bounds
        geom = box(*bounds)
        n = 0
        for row in gdf.iterrows():
            poly = Polygon(row[1]['geometry'])
            if geom.contains(poly):
                n += 1
                name = row[1]['NAME']
        if n == 2:
            name == 'ROSEBUD'
        out_filename = os.path.join(out_directory, name + "_" + year + ".tif")
        print(out_filename)
        # os.rename(raster, out_filename)

def check_for_missing_rasters(rasters, county_shapefile):
    osb = os.path.basename
    years = [str(r) for r in range(2000, 2020)]
    gdf = gpd.read_file(county_shapefile)
    counties = gdf.loc[:, 'NAME']
    for year in years:
        yearly_rasters = [f for f in rasters if year in osb(f)]
        county_rasters = [osb(f)[:osb(f).find('_')] for f in yearly_rasters]
        missing = counties[~counties.isin(county_rasters)]
        print(missing, len(yearly_rasters), counties.shape[0], year)

def get_pairs():
    files = glob('/home/thomas/ssd/montana-image-data/*-*')
    files = [f for f in files if 'xml' not in f]
    pairs = []
    for i in range(len(files)):
        target = os.path.basename(files[i])
        if target[:2] == '24':
            continue
        if 'County20001600091692.376564' in target:
            continue

        target = target[:target.find('-')]
        target = target.replace('5632', '0000')
        for j in range(i+1, len(files)):
            to_match = os.path.basename(files[j])
            to_match = to_match[:to_match.find('-')]
            to_match = to_match.replace('5632', '0000')
            if target == to_match:
                pairs.append([(files[i]), (files[j])])

def clip_raster_to_shapefiles_gdal(raster_file, shapefiles, out_directory, year):

    for f in shapefiles:
        out_filename = os.path.join(out_directory, 
                os.path.splitext(os.path.basename(f))[0] + "_" + year + ".tif")
        if not os.path.isfile(out_filename):
            print('clipping', raster_file, 'to', f, 'saving to', out_filename)
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
        if image_stack.dtype == rasterio.uint16:
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


def plot_df(df, counties=None):
    # bin.
    if counties == None:
        counties = list(df.columns.drop('year'))
    years = df.loc[:, 'year'].astype(np.int32)

    plt.figure(figsize=(18, 15))
    for i, county in enumerate(counties):
        acreage = df.loc[:, county]
        plt.plot(years, acreage, label=county)
        plt.plot(years, acreage, 'k.')
    plt.xticks(years)
    plt.ylabel('irrigated acreage')
    plt.xlabel('year')
    plt.legend()
    plt.title('irr. area, selected counties in MT')


if __name__ == '__main__':
    pass
