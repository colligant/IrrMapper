import warnings
import os
import glob
from  multiprocessing import Pool
from numpy import save as nsave
from compose_array_single_shapefile import PTASingleShapefile, ShapefileSamplePoints
from fiona import open as fopen
from shapely.geometry import shape
from data_utils import download_images, get_shapefile_path_row, split_shapefile, create_master_raster

def download_images_over_shapefile(shapefile, image_directory, year, master_raster_directory):
    '''Downloads p/r corresponding to the location of 
       the shapefile, and creates master raster'''
    p, r = get_shapefile_path_row(shapefile) 
    suff = str(p) + '_' + str(r) + "_" + str(year)
    landsat_dir = os.path.join(image_directory, suff)
    satellite = 8
    if year < 2013:
        satellite = 7
    if not os.path.isdir(landsat_dir):
        os.mkdir(landsat_dir)
        ims = download_images(landsat_dir, p, r, year, satellite)
    else:
        ims = download_images(landsat_dir, p, r, year, satellite)

    ms = create_master_raster(ims, p, r, year, master_raster_directory)
    
    return ims

def create_sample_points_from_shapefile(shapefile_path, instances):
    '''Hopefully this can be nicely parallelized.'''
    ssp = ShapefileSamplePoints(shapefile_path, m_instances=instances)
    ssp.create_sample_points(save_points=True)
    return ssp.outfile 

def shapefile_area(shapefile):
    summ = 0
    with fopen(shapefile, "r") as src:
        for feat in src:
            poly = shape(feat['geometry'])
            summ += poly.area
    return summ

def get_total_area(data_directory, filenames):
    ''' Gets the total area of the polygons
        in the files in filenames
        TODO: Get an equal-area projection'''

    tot = 0
    for f in glob.glob(data_directory + "*.shp"):
        if "sample" not in f:
            for f2 in filenames:
                if f2 in f:
                    tot += shapefile_area(f)
    return tot

def required_points(shapefile, total_area, total_instances):
    area = shapefile_area(shapefile)
    frac = area / total_area
    return int(total_instances * frac)

def split_shapefiles_multiproc(f):
    data_directory = 'split_shapefiles_west/'
    shp_dir = '/home/thomas/IrrigationGIS/western_states_irrgis/western_gis_backup'
    fname = os.path.basename(f) 
    split_shapefile(shp_dir, fname, data_directory)


if __name__ == "__main__":

    image_directory = 'image_data/'
    p = 34
    r = 28
    year = 2013
    satellite = 8
    image_directory += str(p) + "_" + str(r) + "_" + str(year)
    if not os.path.isdir(image_directory):
        os.mkdir(image_directory)

    ims = download_images(image_directory, p, r, year, satellite)

    # irrigated = ['MT_Sun_River_2013', "MT_Huntley_Main_2013"]
    # other = ['other']
    # fallow = ['Fallow']
    # forest = ['Forrest']

    # train_dir = 'training_data/'
    # data_directory = 'split_shapefiles_west/'
    # image_directory = 'image_data/'
    # raster_dir = 'master_rasters'
    # kernel_size = 57

    # fnames = [f for f in glob.glob(data_directory + "*.shp")]

    # instances = 10000
    # instances = [instances]*len(fnames)

    # with Pool() as pool:
    #     out = pool.starmap(create_sample_points_from_shapefile, zip(fnames, instances))


