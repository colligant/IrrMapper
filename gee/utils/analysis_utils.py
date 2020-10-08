import os
import numpy as np
import rasterio
import json
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns

import warnings; warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)

from shapely.geometry import box, Polygon
import runspec as rs
from rasterio import merge
from copy import deepcopy
from glob import glob
from subprocess import check_output
from collections import defaultdict

ACRES_PER_SQUARE_METER = 0.000247105

def crop_proportions_over_time(json_path):
    n = 6
    cdl_crop_values = {**rs.cdl_crop_values(), **rs.cdl_non_crop_values(), 
            **rs.cdl_wateresque_values()}
    with open(json_path, 'r', encoding='utf-8') as f:
        year_to_crop = json.loads(f.read())

    bars = defaultdict(dict)

    for year, crops in year_to_crop.items():
        crops = crops['cropland']
        arr = np.zeros((len(crops), 2))
        for i, (crop, pixels) in enumerate(crops.items()):
            arr[i, 0] = int(crop)
            arr[i, 1] = float(pixels)
        indices = np.argsort(arr[:, 1])[::-1]
        sorted_arr = arr[indices, :]
        top_n = np.zeros((n+1, 2))
        top_n[:n, :] = sorted_arr[:n, :]
        top_n[n, :] = np.sum(sorted_arr[n:, :])
        percentages = top_n[:, 1] / np.sum(top_n[:, 1])
        crop_types = [cdl_crop_values[int(i)] for i in top_n[:-1, 0]]
        print('--------{}--------'.format(year))
        for ty, per in zip(crop_types, percentages):
            # print(ty, per)
            bars[year][ty] = per
        bars[year]['Other'] = percentages[-1]

    f, ax = plt.subplots()
    df = pd.DataFrame.from_dict(bars)
    print(df)
    sns.set()
    x = df.T.plot(kind='bar', stacked=True, ax=ax)
    plt.legend(loc='center left', bbox_to_anchor=(-0.15, 0.9))
    ax.grid(False)
    plt.xticks(rotation=30)
    plt.xlabel('year')
    plt.ylabel('Irrigated proportion')
    plt.show()

     


def merge_rasters_gdal(raster_files, out_filename):
    if not len(raster_files):
        return
    if not os.path.isfile(out_filename):
        print('processing', out_filename)
        cmd = check_output(['gdal_merge.py', '-of', 'GTiff', '-ot', 'Byte',
            '-o', out_filename] + raster_files)
        print(cmd)

def flu_data_irr_area_by_county(county_shp, flu, out_filename, plot=False,
        save=False):

    if os.path.isfile(out_filename):
        return

    flu = gpd.read_file(flu)
    flu = flu.loc[flu['LType'] == 'I']
    counties = gpd.read_file(county_shp)

    flu = flu.to_crs('EPSG:5070')
    counties = counties.to_crs('EPSG:5070')
    counties_with_irr_attr = counties.copy()

    irrigated_area = {}
    for row in counties.iterrows():
        polygon = Polygon(row[1]['geometry'])
        county_name = row[1]['NAME']
        try:
            flu_county = gpd.clip(flu, polygon)
        except Exception as e:
            print('error', county_name, e)
            irrigated_area[county_name] = -1
            continue
        if plot:
            fig, ax = plt.subplots()
            flu_county.plot(ax=ax)
            poly_gdf = gpd.geopandas.GeoDataFrame([1], geometry=[polygon], crs=counties.crs)
            poly_gdf.boundary.plot(ax=ax, color="red")
            plt.title(county_name)
            plt.show()
        else:
            irr_acres = np.sum(flu_county['geometry'].area)
            irrigated_area[county_name] = irr_acres

    names = list(irrigated_area.keys())
    areas = list(irrigated_area.values())
    counties_with_irr_attr['IRR_AREA'] = areas
    if save:
        counties_with_irr_attr.to_file(out_filename)


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

        if n > 1:
            raise ValueError("raster {} contains more than one county".format(raster))
        else:
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


def clip_raster_to_shapefiles_gdal(raster_file, shapefiles, out_directory, year):

    for f in shapefiles:
        out_filename = os.path.join(out_directory, 
                os.path.splitext(os.path.basename(f))[0] + "_" + year + ".tif")
        if not os.path.isfile(out_filename):
            print('clipping', raster_file, 'to', f, 'saving to', out_filename)
            cmd = check_output(['gdalwarp', '-of', 'GTiff', '-cutline', f, 
                               '-crop_to_cutline', raster_file, out_filename])

def get_irrigated_statistics(rasters_by_county, csv_out):

    county_to_year_and_acres = defaultdict(dict)

    for i, raster in enumerate(rasters_by_county):
        ss = os.path.splitext((os.path.basename(raster)))[0]
        year = ss[-4:]
        name = ss[:-5]
        county_to_year_and_acres[name][year] = au.calc_irr_area(raster)
        print(i)

    irr = pd.DataFrame.from_dict(county_to_year_and_acres)
    irr = irr.sort_index() # sort by year
    irr = irr.sort_index(axis=1) # and county name
    irr.to_csv(csv_out)

def tabulate_flu_data(shapefiles, out_filename):

    county_to_year_and_acres = defaultdict(dict)
    for shp in shapefiles:
        year = oss(osb(shp))[0][-4:]
        flu = gpd.read_file(shp)
        for i, county in flu.iterrows():
            name = county['NAME'].lower().replace(" ", "_")
            area = county['IRR_AREA']
            county_to_year_and_acres[name][year] = area

    df = pd.DataFrame.from_dict(county_to_year_and_acres)
    df = df.sort_index()
    df = df.sort_index(axis=1)
    df.to_csv(out_filename)


def calc_irr_area(f):
    if not isinstance(f, np.ndarray):
        with rasterio.open(f, 'r') as src:
            arr = src.read()
    else:
        arr = f
    amax = np.argmax(arr, axis=0)
    mask = (np.sum(arr, axis=0) == 0)
    amax = amax[~mask]
    irrigated = amax[amax == 0]
    return irrigated.shape[0]*(30**2)*ACRES_PER_SQUARE_METER


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

def precip_timeseries():
    irr = pd.read_csv('/home/thomas/mt/statistics/irrigated_acreage_cnn_sept28.csv')
    irr_summed = np.sum(irr, axis=1)
    precip = pd.read_csv('/home/thomas/mt/statistics/precip_1999_2020.csv')
    precip['date'] = [int(str(p)[:-2]) for p in precip['date']]
    summ = pd.DataFrame(irr_summed)
    summ['date'] = np.arange(2000, 2020)
    summ = summ.rename(columns={0:'acreage'})

    sns.set()
    fig, ax = plt.subplots()
    precip.plot(x='date', y='precip', ax=ax, label='precip (in)', legend=False)
    ax.set_ylabel('precip (in)')
    ax1 = ax.twinx()
    ax.set_ylim([14, 23])
    summ.plot(x='date', y='acreage', ax=ax1, c='r', label='irr. area', legend=False)
    ax1.set_ylabel('irr. area (acres)')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax1.grid(None)
    plt.title('irr. area and precip by year')
    plt.show()

if __name__ == '__main__':

    # crop_proportions_over_time('./crop_proportions.json')
    alf = pd.read_csv('/home/thomas/mt/statistics/unirrigated_alfalfa.csv')
    unirr = pd.read_csv('/home/thomas/mt/statistics/irrigated_alfalfa.csv')
    alf = alf.set_index('Unnamed: 0')
    unirr = unirr.set_index('Unnamed: 0')
    precip = pd.read_csv('/home/thomas/mt/statistics/precip_1999_2020.csv')
    precip['date'] = [int(str(p)[:-2]) for p in precip['date']]
    precip = precip[precip['date'] > 2007]
    precip = precip[precip['date'] < 2020]
    precip = precip.set_index('date')


    bars = defaultdict(dict)
    for year in alf.index:
        a = alf.loc[year, :]
        bars[year]['irr. alfalfa'] = np.sum(a)
        bars[year]['unirr. alfalfa'] = np.sum(unirr.loc[year, :])


    sns.set()
    f, ax = plt.subplots()
    df = pd.DataFrame.from_dict(bars)
    df.T.plot(kind='bar', stacked=True, ax=ax)
    plt.xticks(rotation=30)
    ax1 = ax.twinx()
    ax1.plot(range(12), precip['precip'], 'r', label='precip')
    ax1.legend()
    ax1.set_ylabel('precip (in)')
    ax.set_ylabel('irr. area (acres)')
    ax1.grid(None)
    plt.xlabel('year')
    plt.show()
