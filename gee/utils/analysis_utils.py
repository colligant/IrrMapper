import os
import numpy as np
import rasterio
import rasterio.mask
import json
import fiona
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import pickle

import warnings; warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)

from sklearn.metrics import confusion_matrix
from shapely.geometry import box, Polygon
from pyproj import CRS
from rasterio import merge
from copy import deepcopy
from glob import glob
from subprocess import check_output
from collections import defaultdict

import runspec as rs

ACRES_PER_SQUARE_METER = 0.000247105
MONTANA_SHAPEFILE = '/home/thomas/irrigated-training-data-aug21/aux-shapefiles/mt.shp'

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
        county_to_year_and_acres[name][year] = calc_irr_area(raster)
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
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes)
    ax1.grid(None)
    plt.title('irr. area and precip by year')
    plt.show()

def mask_raster_to_shapefile(shapefile, raster, year, return_binary=True):
    ''' 
    Generates a mask with 1 everywhere 
    shapefile data is present and a no_data value everywhere else.
    no_data is -1 in this case, as it is never a valid class label.
    Switching coordinate reference systems is important here, or 
    else the masking won't work.
    '''
    shp = gpd.read_file(shapefile)
    shp = shp[shp.geometry.notnull()]
    with rasterio.open(raster, 'r') as src:
        # pyproj deprecated the +init syntax.
        crs = CRS(src.crs['init'])
        shp = shp.to_crs(crs)
        features = get_features(shp, year)
        if len(features):
            out_image, out_transform = rasterio.mask.mask(src, shapes=features,
                    filled=False)
        else:
            return None, None
        if return_binary:
            out_image[out_image != 0] = 1 
        meta = src.meta
    return out_image, meta

def get_features(gdf, year):
    tmp = json.loads(gdf.to_json())
    if year is not None:
        for feature in tmp['features']:
            break
        features = [feature['geometry'] for feature in tmp['features'] if feature['properties']['YEAR'] == year]
    else:
        features = [feature['geometry'] for feature in tmp['features']]
    return features

def create_class_labels(shapefiles, assign_shapefile_class_code, mask_file, year):
    first = True
    class_labels = None
    for f in shapefiles:
        class_code = assign_shapefile_class_code(f)
        # print(f, class_code)
        osb = os.path.basename(f)
        if 'irrigated' in osb and 'unirrigated' not in osb:
            out, _ = mask_raster_to_shapefile(f, mask_file, year=year, 
                return_binary=False)
        elif 'fallow' in osb:
            out, _ = mask_raster_to_shapefile(f, mask_file, year=year, 
                return_binary=False)
        else:
            out, _ = mask_raster_to_shapefile(f, mask_file, year=None, 
                return_binary=False)

        if out is None:
            print('no features for {}, {}'.format(osb, year))
            continue

        if first:
            class_labels = out
            class_labels[~class_labels.mask] = class_code
            first = False
        else:
            class_labels[~out.mask] = class_code
    return class_labels

def irrigated_label_mask(shapefiles):
    pass


def clip_to_mt_and_get_area(tif, save=False):

    with fiona.open(MONTANA_SHAPEFILE, 'r') as s:
        shapes = [feature['geometry'] for feature in s]

    with rasterio.open(tif, 'r') as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    nbands = out_meta['count']
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    
    out_f = os.path.splitext(tif)[0] + "_clipped.tif"

    if save:
        with rasterio.open(out_f,"w", **out_meta) as dest:
            dest.write(out_image)
    elif nbands == 1:
        print(out_image.shape)

    else:
        amax = np.argmax(out_image, axis=0)
        mask = (np.sum(out_image, axis=0) == 0)
        amax = amax[~mask]
        irrigated = amax[amax == 0]
        return irrigated.shape[0]*(30**2)*ACRES_PER_SQUARE_METER


def filter_shapefile_by_year(shapefile, year):

    outf = os.path.splitext(os.path.basename(shapefile))[0] + "_{}.shp".format(year)
    outf = os.path.join('/tmp/', outf)
    gdf = gpd.read_file(shapefile)
    gdf = gdf.loc[gdf['YEAR'] == year, :]
    if gdf.shape[0] == 0:
        return None
    else:
        return outf

def filter_shapefiles(shapefiles, year):
    shapefiles_ = []
    for s in shapefiles:
        sbs = os.path.basename(s)
        if 'irrigated' in sbs and 'unirrigated' not in sbs:
            shapefiles_.append(filter_shapefile_by_year(s, year))
        elif 'fallow' in sbs:
            shapefiles_.append(filter_shapefile_by_year(s, year))
        else:
            shapefiles_.append(s)
    return [s for s in shapefiles_ if s is not None]

def assign_shapefile_class_code(f):
    f = os.path.basename(f)
    #print(f)
    if 'irrigated' in f and 'unirrigated' not in f:
        return 0
    elif 'fallow' in f or 'unirrigated' in f:
        return 1
    else:
        return 2

def load_raster(raster_name):
    with rasterio.open(raster_name, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta

def prf_from_cmat(cmat):
    tn, fp, fn, tp = cmat.ravel()
    oa = (tn + tp) / (tn + fp + tp + fn)
    prec = (tp)/(tp+fp)
    rec = (tp)/(tp+fn)
    return oa, prec, rec, 2*prec*rec/(prec+rec)

def cmats_from_preds():
    years = [2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015]

    shapefiles = [   'fallow_test.shp',
                     'unirrigated_test.shp',
                     'irrigated_test.shp',
                     'uncultivated_test.shp',
                     'wetlands_buffered_test.shp']

    mask_raster = './irr_median2017.tif'
    root = '/home/thomas/irrigated-training-data-aug21/ee-dataset/data/test/'
    shapefiles = [root + s for s in shapefiles]
    median = False
    median_3 = True
    ocmat = np.zeros((3,3))
    for year in years:
        print(year)
        labels = create_class_labels(shapefiles,
                                     assign_shapefile_class_code,
                                     mask_raster,
                                     year)
        r = '/home/thomas/mt/'
        if median:
            median_raster,_ = load_raster(r + 'bootstrapped/irrmedian{}.tif'.format(year))
            median_raster[median_raster >= np.max(median_raster)/2] = 1
            median_raster[median_raster != 1] = 0
            median_raster = median_raster[~labels.mask]
            labels = labels[~labels.mask]
            cmat = confusion_matrix(labels, median_raster)
            cmat = confusion_matrix(labels, median_raster)
            ocmat += cmat
            print(prf_from_cmat(cmat))
            print(cmat)
            print('----')
        elif median_3:
            median_raster,_ = load_raster('/home/thomas/ssd/median_rasters/irrmedian3bands{}.tif'.format(year))
            amax = np.argmax(median_raster, axis=0)
            # amax[amax == 2] = 1 
            # amax[amax == 0] = 2
            # amax[amax == 1] = 0
            # amax[amax == 2] = 1
            labels = labels.squeeze()
            nodata = np.sum(labels, axis=0) == 0
            median_raster = amax[~labels.mask]
            labels = labels[~labels.mask]
            cmat = confusion_matrix(labels, median_raster)
            ocmat += cmat
            #print(prf_from_cmat(cmat))
            print(cmat)
            print('----')
        else:
            mean_raster,_ = load_raster(r + 'bootstrapped/irrmean{}.tif'.format(year))
            amax = np.argmax(mean_raster, axis=0)
            amax[amax == 2] = 1 
            amax[amax == 0] = 2
            amax[amax == 1] = 0
            amax[amax == 2] = 1
            mean_raster = amax[~labels.squeeze().mask]
            labels = labels[~labels.mask].squeeze()
            cmat = confusion_matrix(labels, mean_raster)
            ocmat += cmat
            print(prf_from_cmat(cmat))
            print(cmat)
            print('----')

    print('final')
    print(ocmat)
    tn, fp, fn, tp = ocmat.ravel()
    oa = (tn + tp) / (tn + fp + tp + fn)
    prec = (tp)/(tp+fp)
    rec = (tp)/(tp+fn)
    print(oa, prec, rec, 2*prec*rec/(prec+rec))
    print('----')




if __name__ == '__main__':

    raster = '/tmp/clip/county.tif'
    roads = '/tmp/clip/roads.shp'
    with rasterio.open(raster, 'r') as src:
        crs = src.crs
        gdf = gpd.read_file(roads)
        features = get_features(gdf.to_crs(crs), None)
        out_image, _ = rasterio.mask.mask(src, features, invert=True, nodata=0)
        arr = src.read()


    print(calc_irr_area(out_image))
    print(calc_irr_area(arr))
    # out_image = np.transpose(out_image, [1,2,0])
    # print(np.max(out_image), np.min(out_image))
    # plt.imshow(out_image)
    # plt.show()
     
    #cmats_from_preds()
    # root = '/home/thomas/ssd/bootstrapped_merged_rasters/'
    # results = os.listdir(root)
    # with open('../mt_counties.pkl', 'rb') as f:
    #     counties = pickle.load(f)
    # counties = '/home/thomas/irrigated-training-data-aug21/aux-shapefiles/MontanaCounties_shp/County.shp'
    # counties = gpd.read_file(counties)
    # counties = counties.to_crs('EPSG:5070')
    # features = json.loads(counties.to_json())

    # model_to_county_and_area = defaultdict(dict)
    # for feature in features['features']:
    #     name = feature['properties']['NAME']
    #     for d in results:
    #         tifs = glob(os.path.join(root, d, "*tif"))
    #         model = d
    #         year_to_area = {}
    #         for tif in tifs:
    #             osb = os.path.splitext(os.path.basename(tif))[0]
    #             year = osb[-4:]
    #             with rasterio.open(tif, 'r') as src:
    #                 out_image, out_transform = rasterio.mask.mask(src,
    #                         [feature['geometry']], crop=True)
    #             area = calc_irr_area(out_image)
    #             year_to_area[year] = [area]
    #         model_to_county_and_area[model][name] = year_to_area
    #     print(name)

    # with open("predictions_by_model.json", 'w') as f:
    #     json.dump(model_to_county_and_area, f)
