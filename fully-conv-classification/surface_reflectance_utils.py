import os
import tarfile 
import datetime
import numpy as np
import pdb
from argparse import ArgumentParser
from multiprocessing import Pool
from glob import glob
from collections import defaultdict
from rasterio import open as rasopen

lsat8_targets = ('band1.tif', 'band2.tif','band3.tif','band4.tif','band5.tif','band6.tif','band7.tif')

lsat7_targets = ('band3.tif', 'band2.tif', 'band1.tif')

def stack_images_from_list_of_filenames_sorted_by_date(filenames):
    first = True
    image_stack = None
    tilesize = 608
    i = 0
    for filename in filenames:
        with rasterio.open(filename, 'r') as src:
            arr = src.read()
            meta = deepcopy(src.meta)
        if first:
            first = False
            image_stack = np.zeros((3*len(filenames), arr.shape[1], arr.shape[2]), 
                    dtype=np.uint16)
            og_meta = deepcopy(meta)
            og_fname = filename
            image_stack[0:3] = arr
            i += 3
        else:
            try:
                image_stack[i:i+3] = arr
                i += 3
            except ValueError as e:
                arr = warp_single_image(filename, og_meta)
                image_stack[i:i+3] = arr
                i += 3
    return image_stack, og_meta, og_fname, meta


def parse_sr_satellite_capture_date_and_path_row(path):
    '''
    LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX
    Where:
    L = Landsat
    X = Sensor (“C”=OLI/TIRS combined, “O”=OLI-only, “T”=TIRS-only, “E”=ETM+, “T”=“TM, “M”=MSS)
    SS = Satellite (”07”=Landsat 7, “08”=Landsat 8)
    LLL = Processing correction level (L1TP/L1GT/L1GS)
    PPP = WRS path
    RRR = WRS row
    YYYYMMDD = Acquisition year, month, day
    yyyymmdd - Processing year, month, day
    CC = Collection number (01, 02, …)
    TX = Collection category (“RT”=Real-Time, “T1”=Tier 1, “T2”=Tier 2)
    '''
    path = os.path.basename(path)
    split = path.split('_') 
    if len(split) < 7:
        raise TypeError("expected filename in LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX format")

    date = datetime.datetime.strptime(split[3], '%Y%m%d')
    path_row = split[2]
    satellite = split[0][2:4]
    if path_row[0] == '_':
        path_row = path_row[1:]
    return satellite, date, path_row




def _maybe_warp(feature_raster, target_geo, target_shape):
    arr, _ = load_raster(feature_raster)
    if arr.shape != target_shape:
        print("#####################")
        print(arr.shape)
        print(feature_raster, target_geo)
        arr = warp_single_image(feature_raster, target_geo)
        print(arr.shape)
    return arr, feature_raster


def _load_rasters(paths_map, target_geo, target_shape):
    single_band = False
    num_rasters = 0
    for key in paths_map:
        if isinstance(paths_map[key], str):
            single_band = True
            num_rasters += 1
        else:
            num_rasters += len(paths_map[key])
    j = 0

    if not single_band:
        feature_rasters = [feature_raster for feat in paths_map.keys() for feature_raster in
        paths_map[feat]]
    else:
        feature_rasters = [paths_map[feat] for feat in paths_map.keys()]
    tg = [target_geo]*len(feature_rasters)
    ts = [target_shape]*len(feature_rasters)
    if not single_band:
        feature_rasters = [feature_raster for feat in paths_map.keys() for feature_raster in
        paths_map[feat]]
    else:
        feature_rasters = [paths_map[feat] for feat in paths_map.keys()]
    tg = [target_geo]*len(feature_rasters)
    ts = [target_shape]*len(feature_rasters)
    with Pool() as pool:
        # Multiprocess the loading of rasters into memory.
        # Speedup of ~40s.
        out = pool.starmap(_maybe_warp, zip(feature_rasters, tg, ts))
    rasters = {feature_raster: array for (array, feature_raster) in out}
    return rasters, num_rasters

def create_image_stack(paths_map):
    first = True
    stack = None
    j = 0
    num_rasters = 0
    for ls in paths_map.values():
        num_rasters += len(ls)
    print(num_rasters)
    for feat in sorted(paths_map.keys()):
        feature_rasters = paths_map[feat]
        for feature_raster in feature_rasters:
            if first:
                arr, target_geo = load_raster(feature_raster)
                stack = np.zeros((num_rasters, arr.shape[1], arr.shape[2]), np.uint16)
                stack[j, :, :] = arr
                j += 1
                first = False
            else:
                try:
                    arr = load_raster(feature_raster)
                    stack[j, :, :] = arr
                    j += 1
                except ValueError:
                    arr = warp_single_image(feature_raster, target_geo)
                    stack[j, :, :] = arr
                    j += 1
    return stack



def stack_rasters_multiprocess(paths_map, target_geo, target_shape):
    first = True
    stack = None
    j = 0
    rasters, num_rasters = _load_rasters(paths_map, target_geo, target_shape)
    for feat in sorted(paths_map.keys()): # ensures the stack is in the same order each time.
        # Ordering within bands is assured by sorting the list that
        # each band corresponding to, as that's sorted by date.
        feature_rasters = paths_map[feat] # maps bands to their location in filesystem.
        for feature_raster in feature_rasters:
            arr = rasters[feature_raster]
            if first:
                stack = np.zeros((num_rasters, target_shape[1], target_shape[2]), np.uint16)
                stack[j, :, :] = arr
                j += 1
                first = False
            else:
                try:
                    stack[j, :, :] = arr
                    j += 1 
                except ValueError as e:
                    print(feature_raster)
                    print(target_geo)
                    raise
    return stack


def stack_rasters_single_scene(paths_map, target_geo, target_shape):
    out = np.zeros((len(paths_map), target_shape[1], target_shape[2]))
    for idx, band_name in enumerate(sorted(paths_map)):
        # rely on sorting to stack
        arr, _ = _maybe_warp(paths_map[band_name], target_geo, target_shape)
        out[idx] = np.squeeze(arr)
    return out


def _select_correct_band(rasters, target_feat):
    for path in rasters:
        if path.endswith(target_feat):
            return rasters[path], path


def stack_rasters(paths_map, target_geo, target_shape):
    first = True
    stack = None
    num_rasters = 0
    for key in paths_map: num_rasters += len(paths_map[key])
    j = 0
    for feat in sorted(paths_map.keys()): # ensures the stack is in the same order each time.
        # Ordering within bands is assured by sorting the list that
        # each band corresponding to, as that's sorting by date.
        feature_rasters = paths_map[feat] # maps bands to their location in filesystem.
        for feature_raster in feature_rasters:
            with rasopen(feature_raster, mode='r') as src:
                arr = src.read()
            if first:
                stack = np.zeros((num_rasters, target_shape[1], target_shape[2]), np.uint16)
                stack[j, :, :] = arr
                j += 1
                first = False
            else:
                try:
                    stack[j, :, :] = arr
                    j += 1
                except ValueError: 
                    arr = warp_single_image(feature_raster, target_geo)
                    stack[j, :, :] = arr
                    j += 1
    return stack


def get_wrs2_features(path, row):

    with fopen(WRS2) as src:
        for feat in src:
            poly = shape(feat['geometry'])
            propr = feat['properties']
            if propr['PATH'] == path and propr['ROW'] == row:
                return [feat]
    return None


def all_rasters(image_directory, satellite=8):
    ''' Recursively get all rasters in image_directory
    and its subdirectories, and adds them to band_map. '''
    band_map = defaultdict(list)
    for band in landsat_rasters()[satellite]:
        band_map[band] = []
    for band in static_rasters():
        band_map[band] = []
    for band in climate_rasters():
        band_map[band] = []

    extensions = (".tif", ".TIF")
    for dirpath, dirnames, filenames in os.walk(image_directory):
        for f in filenames:
            if any(ext in f for ext in extensions):
                for band in band_map:
                    if f.endswith(band):
                        band_map[band].append(os.path.join(dirpath, f))

    for band in band_map:
        band_map[band] = sorted(band_map[band]) # ensures ordering within bands - sort by time.

    return band_map


def _get_path_row_geometry(path, row):
    shp = gpd.read_file(WRS2)
    out = shp[shp['PATH'] == int(path)]
    out = out[out['ROW'] == int(row)]
    return out


def clip_raster(evaluated, path, row, outfile=None):

    out = _get_path_row_geometry(path, row)

    with rasopen(evaluated, 'r') as src:
        out = out.to_crs(src.crs['init'])
        features = get_features(out)
        # if crop == true for mask, you have to update the metadata.
        out_image, out_transform = mask(src, shapes=features, crop=True, nodata=np.nan)
        meta = src.meta.copy()
        count = out_image.shape[0]

    meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
    if outfile is not None:
        save_raster(out_image, outfile, meta, count)


def save_raster(arr, outfile, meta, count=5):
    meta.update(count=count)
    with rasopen(outfile, 'w', **meta) as dst:
        dst.write(arr)


def load_raster(raster_name):
    with rasopen(raster_name, 'r') as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta

prs =  [[34, 26],
        [36, 27],
        [37,28],
        [34,27],
        [39,26],
        [37,29],
        [42,27],
        [41,28],
        [39,29],
        [36,28],
        [40,27],
        [37,26],
        [35,26],
        [38,29],
        [40,28],
        [38,27],
        [35,27],
        [42,26],
        [41,26],
        [40,29],
        [34,29],
        [35,29],
        [38,26],
        [36,26],
        [39,28],
        [41,27],
        [38,28],
        [37,27],
        [36,29],
        [35,28],
        [43,26],
        [39,27],
        [40,26],
        [43,27], 
        [34,28]]

def _parse_landsat_capture_date(landsat_scene):
    '''
    returns: calendar date of scene capture
    landsat_scene is a directory (i.e data/38_27_2013/')
    scene ID:
       LXSPPPRRRYYYYDDDGSIVV
       L = Landsat
       X = Sensor
       S = Satellite
       PPP = WRS Path
       RRR = WRS Row
       YYYY = Year
       DDD = Julian day
       GSI = Ground station ident
       VV = Archived version number
    '''
    julian_year_day = landsat_scene[-10:-5]
    return datetime.datetime.strptime(julian_year_day, '%y%j').date()


def save_image_from_tarball(root, tarball_name, outfile, targets):

    if os.path.isfile(outfile) and 'LE07' not in tarball_name: 
        print(outfile, 'already exists, not recreating')
        return
    print('creating', outfile)

    with tarfile.open(tarball_name, 'r:gz') as tar:   
        band_to_filename = defaultdict(set)
        for member in tar:
            name = member.get_info()['name']
            for target in targets:
                if name.endswith(target):
                    band_to_filename[target] = name
        first = True
        for name in band_to_filename.values():
            if not os.path.isfile(os.path.join(root, name)):
                tar.extract(name, path=root)

        for i, band in enumerate(targets):
            try:
                filename = os.path.join(root, band_to_filename[band])
            except KeyError as e:
                print(e)
                return
            with rasopen(filename, 'r') as src:
                arr = src.read()
                meta = src.meta.copy()
            if first:
                rgb_array = np.zeros((len(targets), arr.shape[1], arr.shape[2]), dtype=meta['dtype'])
                rgb_array[i] = arr
                first = False
            else:
                rgb_array[i] = arr
            meta.update({'count':len(targets)})
            with rasopen(outfile, 'w', **meta) as dst:
                dst.write(rgb_array)
        
        for name in band_to_filename.values():
            if os.path.isfile(os.path.join(root, name)):
                os.remove(os.path.join(root, name))


def get_tarball_satellite_date_and_path_row(tarball_name):

    tarball_name = os.path.basename(tarball_name)
    satellite = tarball_name[:4]
    path_row = tarball_name[4:10]
    date = tarball_name[10:18]
    date = datetime.datetime.strptime(date, '%Y%m%d')
    return satellite, date, path_row


def sort_tarfiles_into_path_row_directories(tarball_directory, dry_run=True):

    path_row_to_tarball = defaultdict(list)
    for f in glob(os.path.join(tarball_directory, "*gz")):
        print('processing', f)
        try:
            _, _, path_row = get_tarball_satellite_date_and_path_row(f)
        except Exception as e:
            print(e)
            continue
        path_row_to_tarball[path_row].append(f)

    for path_row, list_of_tarballs in path_row_to_tarball.items():

        try:
            out_directory = os.path.join(tarball_directory, path_row)
            if not os.path.isdir(out_directory):
                os.mkdir(out_directory)
            for tarball in list_of_tarballs:
                if not dry_run:
                    os.rename(tarball, os.path.join(out_directory, os.path.basename(tarball)))
                else:
                    print('dry run: moving {} to {}'.format(tarball, 
                        os.path.join(out_directory, os.path.basename(tarball))))
        except Exception as e:
            print(e)


path_row_targets = set(['036029', '038029', '038027', '040029', '034029', '040027', 
        '037028', '039027', '040028', '035027', '035029', '041027', '042027', 
        '037029', '040026', '038028', '035028', '034026', '042026', '043026',
        '039026', '043027', '038026', '034027', '036028', '036026', '039028', 
        '039029', '034028', '041026', '036027', '041028', '035026', '037026', '037027'])


def criteria(satellite, date, path_row):

    if '7' in satellite:
        return False
    if date.year != 2013:
        return False
    if path_row not in path_row_targets:
        return False
    return True

path_row_targets = set(['036026', '036027', '037026', '036028', '037028', '034027', '035027',
    '041027', '042027', '041028', '040028', '039026', '038026', '039027', '038027',
    '040027', '039028', '038028', '039039'])


if __name__ == '__main__':
    # organization: 
    # all the data is stored in path/row directories
    # to make rgb images, read directory and make sure dates are the same.
    # 
    ap = ArgumentParser()
    ap.add_argument("--tar-directory", required=True, help='directory where the tarballs are saved')
    ap.add_argument("--out-directory", required=True, help='directory in which the RGB images are saved')
    # save commands in file
    # with something like this...
    #ls /home/thomas/ssd/tartest/ | while read line;
    #do echo -n python data_utils.py "--tar-directory /home/thomas/ssd/tartest/$line --out-directory .";
    # echo
    # done > commands.txt
    # then do parallel < commands.txt
    args = ap.parse_args()
    tar_directory = args.tar_directory
    out_directory = args.out_directory
    
    ## REMEMBER TO DO THE TAR FILES.
    # REMEMBER TO DO THE DIRECTORIES
    
    targets =  lsat8_targets

    fs = os.listdir(tar_directory)
    dirs = [f for f in fs if os.path.isdir(os.path.join(tar_directory, f))]
    dirs = [f for f in dirs if f in path_row_targets]
    tot_gz_files = []
    pr_to_files = defaultdict(list)
    print(len(path_row_targets))
    files = glob(os.path.join(tar_directory, "*gz"))
    root = '/media/synology/stacked_images_2015_mt/'
    for f in files:
        satellite, date, path_row = get_tarball_satellite_date_and_path_row(f)
        path = path_row[1:3]
        row = path_row[4:]
        path_row = '0{}0{}'.format(path,row)
        if (date.year == 2015 and satellite == 'LC08' and path_row in path_row_targets):
            pr_to_files[path_row].append(f)
            tot_gz_files.append(f)
            datestring = str(date.year) + "_" + str(date.month) + "_" + str(date.day)
            unique_filename = os.path.join(out_directory, "image_d{}_p{}.tif".format(datestring,
                path_row))
            try:
                save_image_from_tarball(root, f, unique_filename, targets)
            except Exception as e:
                print(e)

    # print(len(tot_gz_files))
    # print(path_row_targets)

    '''
    for d in dirs:
        for i, f in enumerate(glob(os.path.join(tar_directory, d, "*gz"))):
            try:
                satellite, date, path_row = get_tarball_satellite_date_and_path_row(f)
                path = path_row[1:3]
                row = path_row[4:]
                path_row = '0{}0{}'.format(path,row)
                if not criteria(satellite, date, path_row):
                    continue
                datestring = str(date.year) + "_" + str(date.month) + "_" + str(date.day)
                unique_filename = os.path.join(out_directory, "image_d{}_p{}.tif".format(datestring,
                    path_row))
                root = os.path.join(tar_directory, d)
                save_image_from_tarball(root, f, unique_filename, targets)
            except Exception as e:
                print(e)

    for i, f in enumerate(glob(os.path.join(tar_directory, "*gz"))):
        try:
            satellite, date, path_row = get_tarball_satellite_date_and_path_row(f)
            path = path_row[1:3]
            row = path_row[4:]
            path_row = '{}_{}'.format(path,row)
            print(criteria(satellite, date, path_row))
            continue
            datestring = str(date.year) + "_" + str(date.month) + "_" + str(date.day)
            unique_filename = os.path.join(out_directory, "image_d{}_p{}.tif".format(datestring,
                path_row))
            save_image_from_tarball(tar_directory, f, unique_filename, targets)
        except tarfile.ReadError:
            with open('possibly_bad_archives.txt', 'a') as fil:
                print(f, file=fil)
    #sort_tarfiles_into_path_row_directories(tar_directory, dry_run=False)
    '''
