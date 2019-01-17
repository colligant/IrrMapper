from shapely.geometry import shape
import glob
import os
from collections import defaultdict
import fiona
from lxml import html
from requests import get
from copy import deepcopy
from numpy import zeros
import re
from rasterio import float32, open as rasopen
from prepare_images import ImageStack
import sat_image

WRS_2 = '../spatial_data/wrs2_descending.shp' 

def create_master_masked_raster(image_stack, path, row, year, raster_directory):
    masks = image_stack.masks
    if not masks:
        return None

    first = True
    stack = None

    for i, feat in enumerate(masks.keys()): # ordered dict ensures accuracy here.

        mask_raster = masks[feat] # maps bands to their location in filesystem.

        with rasopen(mask_raster, mode='r') as src:
            arr = src.read()
            raster_geo = src.meta.copy()

        if first:
            first_geo = deepcopy(raster_geo)
            print(first_geo, "FIRST_GEO")
            empty = zeros((len(masks.keys()), arr.shape[1], arr.shape[2]), float32)
            stack = empty
            stack[i, :, :] = arr
            first = False
        else:
            try:
                stack[i, :, :] = arr
            except ValueError: 
                import pprint
                pprint.pprint(first_geo)
                # error was thrown here b/c source raster didn't have crs
                arr = sat_image.warped_vrt.warp_single_image(mask_raster, first_geo)
                stack[i, :, :] = arr

    first_geo.update(count=len(masks.keys()))

    fname = "master_mask_raster_{}_{}_{}.tif".format(path, row, year)
    pth = os.path.join(raster_directory, fname)

    with rasopen(pth, mode='w', **first_geo) as dst:
        dst.write(stack) 

    return pth


def create_master_raster(image_stack, path, row, year, raster_directory):
    fname = "master_raster_{}_{}_{}.tif".format(path, row, year)
    pth = os.path.join(raster_directory, fname)
    if os.path.isfile(pth):
        print("Master raster already created for {}_{}_{}.".format(path, row, year))
        return pth 
        
    paths_map = image_stack.paths_map
    first = True
    stack = None

    for i, feat in enumerate(paths_map.keys()): # ordered dict ensures accuracy here.

        feature_raster = paths_map[feat] # maps bands to their location in filesystem.

        with rasopen(feature_raster, mode='r') as src:
            arr = src.read()
            raster_geo = src.meta.copy()

        if first:
            first_geo = deepcopy(raster_geo)
            empty = zeros((len(paths_map.keys()), arr.shape[1], arr.shape[2]), float32)
            stack = empty
            stack[i, :, :] = arr
            first = False
        else:
            try:
                stack[i, :, :] = arr
            except ValueError: 
                # import pprint
                # pprint.pprint(first_geo)
                # error was thrown here b/c source raster didn't have crs
                arr = sat_image.warped_vrt.warp_single_image(feature_raster, first_geo)
                stack[i, :, :] = arr

    first_geo.update(count=len(paths_map.keys()))

    with rasopen(pth, mode='w', **first_geo) as dst:
        dst.write(stack)

    return pth


def get_path_row(lat, lon):
    """
    :param lat: Latitude float
    :param lon: Longitude float
            'convert_pr_to_ll' [path, row to coordinates]
    :return: lat, lon tuple or path, row tuple
    """
    conversion_type = 'convert_ll_to_pr'
    base = 'https://landsat.usgs.gov/landsat/lat_long_converter/tools_latlong.php'
    unk_number = 1508518830987

    full_url = '{}?rs={}&rsargs[]={}&rsargs[]={}&rsargs[]=1&rsrnd={}'.format(base,
                                                                             conversion_type,
                                                                             lat, lon,
                                                                             unk_number)
    r = get(full_url)
    tree = html.fromstring(r.text)

    # remember to view source html to build xpath
    # i.e. inspect element > network > find GET with relevant PARAMS
    # > go to GET URL > view source HTML
    p_string = tree.xpath('//table/tr[1]/td[2]/text()')
    path = int(re.search(r'\d+', p_string[0]).group())

    r_string = tree.xpath('//table/tr[1]/td[4]/text()')
    row = int(re.search(r'\d+', r_string[0]).group())

    return path, row

def get_shapefile_lat_lon(shapefile):
    with fiona.open(shapefile, "r") as src:
        minx, miny, maxx, maxy = src.bounds
        latc = (maxy + miny) / 2
        lonc = (maxx + minx) / 2

    return latc, lonc 


def download_images(project_directory, path, row, year, satellite=8):

    image_stack = ImageStack(satellite=satellite, path=path, row=row, root=project_directory,
            max_cloud_pct=70, n_landsat=3, year=year)

    image_stack.build_evaluating() # the difference b/t build_training() and build_eval() is
    # a cloud mask.
    return image_stack


def get_pr(poly, wrs2):
    ls = []
    for feature in wrs2:
        tile = shape(feature['geometry'])
        if poly.within(tile):
            z = feature['properties']
            p = z['PATH']
            r = z['ROW']
            ls.append(str(p) + "_" + str(r))
    return ls

def split_shapefile(base, base_shapefile, data_directory):
    """ Shapefiles may deal with data over multiple path/rows.
    Data directory: where the split shapefiles will be saved."""
    path_row = defaultdict(list) 
    id_mapping = {}
    wrs2 = fiona.open('../spatial_data/wrs2_descending_usa.shp', 'r')

    with fiona.open(os.path.join(base, base_shapefile), "r") as src:
        meta = deepcopy(src.meta)
        for feat in src:
            idd = feat['id']
            id_mapping[idd] = feat
            poly = shape(feat['geometry'])
            prs = get_pr(poly, wrs2)
            for p in prs:
                path_row[p].append(idd)

    wrs2.close()

    # I have all path/rows and their corresponding features
    # I need to figure out the unique features in each path row.
    # How should I treat the non-unique features?
    # Create a set of non-unique features and their 
    # corresponding path/row. Also create
    # a set of unique features. Then iterate over the 
    # unique set and for each non-unique feature 
    # place it in the path/row with the greatest number of
    # unique points. 
    non_unique = defaultdict(list)
    unique = defaultdict(list)
    for key in path_row:
        ls = path_row[key] # all features in a given path/row
        placeholder = ls
        for key1 in path_row:
            if key != key1:
                ls1 = path_row[key1]
                # find unique keys in ls
                placeholder = set(placeholder) - set(ls1) #all 
                # features present in placeholder that are not 
                # present in ls1; i.e. unique keys
        unique[key] = list(placeholder)
        nu = set(ls) - set(placeholder) # all features present
        # in ls that are not present in placeholder (non-unique)
        for idd in list(nu):
            non_unique[idd].append(key)
   
    for key in non_unique: # unique ids 
        pr = None 
        hi = 0
        for pathrow in non_unique[key]: # path/rows corresponding to non
            # unique features
            if len(unique[pathrow]) > hi:
                pr = pathrow 
                hi = len(unique[pathrow])
        if pr is not None:
            unique[pr].append(key)

    prefix = os.path.splitext(base_shapefile)[0]
    for key in unique:
        out = prefix + "_" + key + ".shp"
        if len(unique[key]):
            with fiona.open(os.path.join(data_directory, out), 'w', **meta) as dst:
                print("Split shapefile saving to:", 
                        os.path.join(data_directory, out))
                for feat in unique[key]:
                    dst.write(id_mapping[feat])

def get_shapefile_path_row(shapefile):
    """This function assumes that the original
    shapefile has already been split, and relies on
    the naming convention to get the path and row.  """
    # strip extension
    # TODO: Find some way to update shapefile metadata
    shp = shapefile[-9:-4].split("_")
    return int(shp[0]), int(shp[1])
    
if __name__ == "__main__":
    pass
    # base = "/home/thomas/IrrigationGIS/western_states_irrgis/MT/"
    # base_shapefile = "MT_Huntley_Main_2013_3728.shp"
    # data_directory = "shapefile_data/"
    # split_shapefile(base, base_shapefile, data_directory)
