from lsru import Usgs, Espa, Order, UnavailableDataProductException
from lsru.utils import bounds, geom_from_metadata
from ast import literal_eval
import pdb
import geopandas as gpd

from fiona import open as fopen
from shapely.geometry import shape
from datetime import datetime
from data_utils import prs

usgs = Usgs()
usgs.login()

espa = Espa()
unique_gdf = gpd.read_file("/home/thomas/irrigated-training-data/data/agg/aggregated_target_path_rows.shp")

targ = [42026.0, 40029.0,35029.0,43026.0,43027.0]
good_idx = unique_gdf['PR'].isin(targ)

unique_gdf = unique_gdf.loc[good_idx, :]
with open('./completed_scenes.txt', 'r') as src:
    ls = src.read()
    ls = ls.split('\n')
completed_scenes = set(ls)
for index, data_row in unique_gdf.iterrows():
    scene_lists = []
    year = int(data_row['YEAR'])
    for year in [2013]:
        print('ordering', year, data_row['PR'])
        feature = shape(data_row['geometry'])
        centroid = feature.centroid
        bbox = centroid.buffer(0.0001).bounds
        scene_list = usgs.search(collection='LANDSAT_8_C1',
                                 bbox=bbox,
                                 begin=datetime(year, 1, 1),
                                 end=datetime(year + 1, 1, 1),
                                 max_results=10000,
                                 max_cloud_cover=100)
        scene_lists.extend(scene_list)
        #cname = usgs.get_collection_name(7)
        #scene_list = usgs.search(collection=cname,
        #                         bbox=bbox,
        #                         begin=datetime(year, 1, 1),
        #                         end=datetime(year + 1, 1, 1),
        #                         max_results=10000,
        #                         max_cloud_cover=100)
        #scene_lists.extend(scene_list)
        display_ids = [x['displayId'] for x in scene_lists]
        uncompleted_scenes = []
        for scene in display_ids:
            if scene not in completed_scenes:
                uncompleted_scenes.append(scene)
        print(len(uncompleted_scenes))
        if len(uncompleted_scenes):
            try:
                order = espa.order(uncompleted_scenes, products=['sr'])
            except UnavailableDataProductException as e:
                print(e)
                continue
                exception_dict = e.args[0][0]
                list_of_bad_products = exception_dict['2 validation errors'][0]
                list_of_bad_products = list_of_bad_products[list_of_bad_products.find('['):list_of_bad_products.find(']')+1]
                print(list_of_bad_products)
                list_of_bad_products = literal_eval(list_of_bad_products)

                for ll in list_of_bad_products:
                    uncompleted_scenes.remove(ll)
                order = espa.order(uncompleted_scenes, products=['sr'])
            with open('completed_scenes.txt', 'a') as f:
                for s in uncompleted_scenes:
                    print(s, file=f)

# print(len(scene_lists))
# # Extract Landsat scene ids for each hit from the metadata
# 
# scene_list = [x['displayId'] for x in scene_lists]
# print(scene_list)

#scene_list = list(scene_list)
#order = espa.order(scene_list, products=['sr'])

#p = '/home/thomas/share/surface-reflectance/'
#print(len(espa.orders))
#comp = [order.is_complete for order in espa.orders]
#print(sum(comp))
#for order in espa.orders:
#    if order.is_complete:
#        order.download_all_complete(p, check_complete=True)
