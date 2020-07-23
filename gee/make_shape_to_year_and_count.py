import os
import geopandas as gpd
from glob import glob
from collections import defaultdict
from pprint import pprint

YEARS = [2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015]

def is_temporal(shapefile_path):
    shapefile_path = os.path.basename(shapefile_path)
    if 'fallow' in shapefile_path:
        return True
    elif 'irrigated' in shapefile_path and 'unirrigated' not in shapefile_path:
        return True
    else:
        return False

def make_dict(shapefile_dir):
    shape_to_year_and_count = defaultdict(dict)

    for f in glob(os.path.join(shapefile_dir, "*shp")):
        gdf = gpd.read_file(f)
        stripped_filename = os.path.splitext(os.path.basename(f))[0]
        for year in YEARS:
            if is_temporal(f):
                shape_to_year_and_count[stripped_filename][year] = sum(gdf['YEAR'] == year)
            else:
                shape_to_year_and_count[stripped_filename][year] = gdf.shape[0]

    return shape_to_year_and_count

if __name__ == '__main__':


    shapefile_train = '/home/thomas/irrigated-training-data/ee-dataset/train/'
    shapefile_test = '/home/thomas/irrigated-training-data/ee-dataset/test/'
    shapefile_valid = '/home/thomas/irrigated-training-data/ee-dataset/validation/'

    train = make_dict(shapefile_train)
    test = make_dict(shapefile_test)
    valid = make_dict(shapefile_valid)
    merged = {**train, **test, **valid} # wow python

    pprint(merged)


