import os
import analysis_utils as au
from glob import glob
import pandas as pd
import geopandas as gpd
from collections import defaultdict


osb = os.path.basename
osj = os.path.join
oss = os.path.splitext

if __name__ == '__main__':


    fname = '/home/thomas/mt/statistics/irrigated_acreage_flu_sept28.csv'
    shapefiles = glob('/home/thomas/flu-data/*irrigated_area*.shp')
    dct = defaultdict(dict)
    for f in glob("/home/thomas/ssd/rasters_clipped_to_counties/*tif"):
        ff = oss(osb(f))[0]
        year = ff[-4:]
        name = ff[:-5]
        dct[name][year] = au.calc_irr_area(f)

    df = pd.DataFrame.from_dict(dct)
    df = df.sort_index()
    df = df.sort_index(axis=1)
    df.to_csv('/home/thomas/mt/statistics/irrigated_acreage_cnn_sept28.csv')
