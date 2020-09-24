import os
import analysis_utils as au
from glob import glob

osb = os.path.basename
 
if __name__ == '__main__':
    # root = '/home/thomas/irrigated-training-data-aug21/aux-shapefiles/'
    # shapefile = root + 'MontanaCounties_shp/County.shp'

    # rasters = glob('/home/thomas/montana_data/montana-image-data-2016-and-on/*tif')
    # rasters += glob('/home/thomas/ssd/montana-image-data/*tif')
    # out_directory = '/home/thomas/montana_data/montana-image-data-2016-and-on/'
    # au.check_for_missing_rasters(rasters, shapefile)

    rasters = glob('/home/thomas/mt/montana-irr-rasters/rasters_by_county/*tif')
    years =[str(y) for y in range(2000, 2020) ]
    for year in years: 
        yearly_rasters = [r for r in rasters if year in r]
        out = '/home/thomas/mt/montana-irr-rasters/rasters/irrMT_{}.tif'.format(year)
        au.merge_rasters_gdal(yearly_rasters, year)
