import ee
# ee.Authenticate()
ee.Initialize()
import time
import os
import numpy as np
from random import shuffle

import utils.ee_utils as ee_utils
from utils.shapefile_spec import shape_to_year_and_count as SHP_TO_YEAR_AND_COUNT

class GEEExtractor:

    def __init__(self, year, out_gs_bucket, out_folder, mask_shapefiles, kernel_size=256,
            n_shards=10):

        self.year = year
        self.iter = 0
        self.out_folder = out_folder
        self.out_gs_bucket = out_gs_bucket
        self.n_shards = n_shards
        self.mask_shapefiles = mask_shapefiles
        self.kernel_size = kernel_size

        self._construct_data_stack()

    
    def _construct_data_stack(self):
        image_stack = ee_utils.preprocess_data(self.year).toBands()
        features = [feat['id'] for feat in image_stack.getInfo()['bands']]
        self.features = features + ['elevation', 'slope', 'constant']  

        self.shapefile_to_feature_collection = \
                ee_utils.temporally_filter_features(self.mask_shapefiles, self.year)

        class_labels = ee_utils.create_class_labels(self.shapefile_to_feature_collection)

        terrain = ee.Image("USGS/SRTMGL1_003").select('elevation')
        slope = ee.Terrain.slope(terrain)

        self.image_stack = ee.Image.cat([terrain, slope]).float()
        # self.image_stack = ee.Image.cat([image_stack, terrain, slope, class_labels]).float()

        list_ = ee.List.repeat(1, self.kernel_size)
        lists = ee.List.repeat(list_, self.kernel_size)
        kernel = ee.Kernel.fixed(self.kernel_size, self.kernel_size, lists)
        self.data_stack = self.image_stack.neighborhoodToArray(kernel)

        self.projection = ee.Projection('EPSG:5070')
        self.data_stack = self.data_stack.reproject(self.projection, None, 30)
        self.image_stack = self.image_stack.reproject(self.projection, None, 30)


    def extract_data_over_patches(self, patch_shapefiles, target_patch_name=None,
            buffer_region=None, geotiff=False):
        '''
        Extracts TFRecords over a ROI. ROIs are features in patch_shapefile.
        '''
        if isinstance(patch_shapefiles, list):
            for patch_shapefile in patch_shapefiles:
                patch_shapefile = ee.FeatureCollection(patch_shapefile)
                patches = patch_shapefile.toList(patch_shapefile.size())
                out_filename = self._create_filename(patch_shapefiles)
                for idx in range(patches.size().getInfo()):
                    patch = ee.Feature(patches.get(idx))
                    if buffer_region is not None:
                        patch = patch.buffer(buffer_region)
                    self._create_and_start_image_task(patch, out_filename, idx)
        else:
            patches = ee.FeatureCollection(patch_shapefiles)
            patches = patches.toList(patches.size())
            out_filename = self._create_filename(patch_shapefiles)
            for idx in range(patches.size().getInfo()):
                patch = ee.Feature(patches.get(idx))
                if buffer_region is not None:
                    patch = patch.buffer(buffer_region)
                    patch = ee.Feature(patch.bounds())
                self._create_and_start_image_task(patch, out_filename,
                        idx=idx,
                        target_patch_name=target_patch_name,
                        geotiff=geotiff)


    def extract_data_over_shapefile(self, shapefile, percent=None, num=None,
            shuffle=True):
        '''
        This method samples the data stack over the features in 
        the shapefile that is passed in. Sampling consists
        of choosing a random kernel_sizexkernel_size tile from
        the interior of a feature in the shapefile that is passed in.
        Percent governs the number of features chosen to extract over.
        '''
        try:
            feature_collection = self.shapefile_to_feature_collection[shapefile]
        except KeyError as e:
            feature_collection = ee.FeatureCollection(shapefile)
        feature_collection = feature_collection.toList(feature_collection.size())
        try:
            n_features = SHP_TO_YEAR_AND_COUNT[os.path.basename(shapefile)][self.year]
        except:
            n_features = 5238

        out_filename = self._create_filename(shapefile)

        if percent is not None:
            n = int(n_features * percent / 100)
            if shuffle:
                indices = [int(i) for i in np.random.choice(n_features, size=n, replace=False)]
            else:
                indices = [int(i) for i in range(n)]
        elif num is not None:
            n = int(num)
            assert( n <= n_features )
            if shuffle:
                indices = [int(i) for i in np.random.choice(n_features, size=n, replace=False)]
            else:
                indices = [int(i) for i in range(n)] # GEE wants ints.
        else:
            print("Either percent or num features to extract needs to be specified")
            exit(1)

        if len(indices):
            print('extracting data for {}, with {}/{} features chosen'.format(out_filename, n,
                n_features))
        else:
            print('No features for {}'.format(out_filename))
            return

        geometry_sample = ee.ImageCollection([])
        feature_count = 0
        for idx in indices:
            feat = ee.Feature(feature_collection.get(idx))
            sample = self.data_stack.sample(
                     region=feat.geometry(),
                     scale=30,
                     numPixels=1,
                     tileScale=2,
                     dropNulls=False
                     )
            geometry_sample = geometry_sample.merge(sample)
            if (feature_count+1) % self.n_shards == 0:
                geometry_sample = self._create_and_start_table_task(geometry_sample,
                        out_filename, idx)
            feature_count += 1
        # take care of leftovers
        self._create_and_start_table_task(geometry_sample, out_filename, idx)


    def _create_and_start_image_task(self, patch, out_filename, idx, target_patch_name,
            geotiff=False):

        if target_patch_name is not None:
            if patch.get('NAME').getInfo() not in set(target_patch_name):
                return

        kwargs = {
                'image':self.image_stack,
                'bucket':self.out_gs_bucket,
                'description':out_filename,
                'fileNamePrefix':os.path.join(self.out_folder, out_filename + "_" + str(self.year)\
                        + str(time.time())),
                'crs':'EPSG:5070',
                'region':patch.geometry(),
                'scale':30,
                }

        if geotiff:
            kwargs['fileFormat'] = 'GeoTIFF'

        else:

            kwargs['fileFormat'] = 'TFRecord'
            kwargs['formatOptions'] = {'patchDimensions':256,
                                      'compressed':True,
                                      'maskedThreshold':0.99}

        task = ee.batch.Export.image.toCloudStorage(
                **kwargs
                )
        
        self._start_task_and_handle_exception(task)

    
    def _create_and_start_table_task(self, geometry_sample, out_filename, idx):
        task = ee.batch.Export.table.toCloudStorage(
                collection=geometry_sample,
                description=out_filename + str(time.time()),
                bucket=self.out_gs_bucket,
                fileNamePrefix=self.out_folder + str(idx) + out_filename + str(time.time()),
                fileFormat='TFRecord',
                selectors=self.features
                )
        geom = self._start_task_and_handle_exception(task)
        return geom


    def _start_task_and_handle_exception(self, task):

        try:
            task.start()
        except ee.ee_exception.EEException as e:
            print(e)
            print('waiting to export, sleeping for 50 minutes')
            time.sleep(3000)
            task.start()
        return ee.ImageCollection([])

    def _create_filename(self, shapefile):

        return os.path.basename(shapefile) + str(self.year)



if __name__ == '__main__':

    gs_bucket = 'ee-irrigation-mapping'
    test_root = 'users/tcolligan0/test-data-aug24/'
    # Removed fallow.
    test_shapefiles = ['irrigated_test', 'uncultivated_test', 'unirrigated_test',
            'wetlands_buffered_test']
    test_shapefiles = [test_root + s for s in test_shapefiles]

    train_root = 'users/tcolligan0/train-data-aug24/'
    train_shapefiles = ['irrigated_train', 'uncultivated_train', 'unirrigated_train',
                        'wetlands_buffered_train']
    train_shapefiles = [train_root + s for s in train_shapefiles]
    # really terrible name. It's train points.
    train_points = ['users/tcolligan0/test_points_maybepoints']

# 2003
# 2008
# 2009
# 2010
# 2011
# 2012
# 2013
# 2015
    year = 2013
    extractor = GEEExtractor(year, 
                             out_gs_bucket=gs_bucket, 
                             out_folder='slope-and-dem-oct29/', 
                             mask_shapefiles=train_shapefiles,
                             n_shards=100)
    extractor.extract_data_over_patches('users/tcolligan0/County', geotiff=True)
    '''
    extract_test = True
    extract_train = True
    if extract_train:
        #years = [2008, 2009, 2010, 2011, 2012, 2015]
        years = [2003]
        for year in years:
            extractor = GEEExtractor(year, 
                                     out_gs_bucket=gs_bucket, 
                                     out_folder='train-data-oct28/', 
                                     mask_shapefiles=train_shapefiles,
                                     n_shards=100)
            for shapefile in train_points:
                extractor.extract_data_over_shapefile(shapefile, percent=100, shuffle=False)
    if extract_test:
        #years = [2003, 2008, 2009, 2010, 2011, 2012, 2015]
        years = [2013]
        for year in years:
            extractor = GEEExtractor(year, 
                                     out_gs_bucket=gs_bucket, 
                                     out_folder='test-data-oct28/', 
                                     mask_shapefiles=test_shapefiles,
                                     n_shards=100)
            extractor.extract_data_over_patches('users/tcolligan0/test-data-aug24/test_regions')
    '''
