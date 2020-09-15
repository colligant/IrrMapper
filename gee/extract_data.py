import ee
# ee.Authenticate()
ee.Initialize()
import tensorflow as tf
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
        self.features = features + ['constant']
        self.shapefile_to_feature_collection = \
                ee_utils.temporally_filter_features(self.mask_shapefiles, self.year)
        class_labels = ee_utils.create_class_labels(self.shapefile_to_feature_collection)
        self.image_stack = ee.Image.cat([image_stack, class_labels]).int16()

        list_ = ee.List.repeat(1, self.kernel_size)
        lists = ee.List.repeat(list_, self.kernel_size)
        kernel = ee.Kernel.fixed(self.kernel_size, self.kernel_size, lists)
        self.data_stack = self.image_stack.neighborhoodToArray(kernel)

        self.projection = ee.Projection('EPSG:5070')
        self.data_stack = self.data_stack.reproject(self.projection, None, 30)
        self.image_stack = self.image_stack.reproject(self.projection, None, 30)


    def extract_data_over_patches(self, patch_shapefiles, buffer_region=None):
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
                    self._create_and_start_image_task(patch, out_filename)
        else:
            patches = ee.FeatureCollection(patch_shapefiles)
            patches = patches.toList(patches.size())
            out_filename = self._create_filename(patch_shapefiles)
            for idx in range(patches.size().getInfo()):
                patch = ee.Feature(patches.get(idx))
                if buffer_region is not None:
                    patch = patch.buffer(buffer_region)
                    patch = ee.Feature(patch.bounds())
                self._create_and_start_image_task(patch, out_filename)


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
        n_features = SHP_TO_YEAR_AND_COUNT[os.path.basename(shapefile)][self.year]
        out_filename = self._create_filename(shapefile)

        if percent is not None:
            n = int(n_features * percent / 100)
            if shuffle:
                indices = [int(i) for i in np.random.choice(n_features, size=n, replace=False)]
            else:
                indices = [int(i) for i in np.range(n)]
        elif num is not None:
            n = int(num)
            assert( n <= n_features )
            if shuffle:
                indices = [int(i) for i in np.random.choice(n_features, size=n, replace=False)]
            else:
                indices = [int(i) for i in np.range(n)]
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
            sample = self.data_stack.sample(
                     region=ee.Feature(feature_collection.get(idx)).geometry(),
                     scale=30,
                     numPixels=1,
                     tileScale=8
                     )
            geometry_sample = geometry_sample.merge(sample)
            if (feature_count+1) % self.n_shards == 0:
                geometry_sample = self._create_and_start_table_task(geometry_sample,
                        out_filename)
            feature_count += 1
        # take care of leftovers
        self._create_and_start_table_task(geometry_sample, out_filename)


    def _create_and_start_image_task(self, patch, out_filename):
        county_name = patch.get('NAME').getInfo()
        self.iter += 1 
        task = ee.batch.Export.image.toCloudStorage(
                image=self.image_stack,
                bucket=self.out_gs_bucket,
                description=out_filename + str(time.time()),
                fileNamePrefix=os.path.join(self.out_folder, str(self.iter) + "_" + str(self.year)),
                fileFormat='GeoTIFF',
                crs='EPSG:5070',
                region=patch.geometry(),
                scale=30,
                # formatOptions={'patchDimensions':256,
                #                'compressed':True,
                #                'maskedThreshold':0.99},
                )
        self._start_task_and_handle_exception(task)

    
    def _create_and_start_table_task(self, geometry_sample, out_filename):
        task = ee.batch.Export.table.toCloudStorage(
                collection=geometry_sample,
                description=out_filename + str(time.time()),
                bucket=self.out_gs_bucket,
                fileNamePrefix=self.out_folder + out_filename + str(time.time()),
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
    test_shapefiles = ['irrigated_test', 'uncultivated_test', 'unirrigated_test',
            'wetlands_buffered_test', 'fallow_test']
    test_shapefiles = [test_root + s for s in test_shapefiles]

    train_root = 'users/tcolligan0/train-data-aug24/'
    train_shapefiles = ['irrigated_train', 'uncultivated_train', 'unirrigated_train',
                        'wetlands_buffered_train', 'fallow_train']
    train_shapefiles = [train_root + s for s in train_shapefiles]

    
    years = [2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015]

    years = [2009, 2010, 2011, 2012, 2013]

    patches = 'users/tcolligan0/test-data-aug24/test_regions'
    counties = 'users/tcolligan0/County'
    extract_test = False
    extract_train = True
    
    # need to finish 2010
    for year in range(2011, 2020):
        print('extracting for', year)
        extractor = GEEExtractor(year, 
                                 out_gs_bucket=gs_bucket, 
                                 out_folder='mt_counties_past_2011',
                                 mask_shapefiles=test_shapefiles + train_shapefiles,
                                 n_shards=100)
        extractor.extract_data_over_patches(counties)

    '''


    if extract_test:
        for year in years:
            extractor = GEEExtractor(year, 
                                     out_gs_bucket=gs_bucket, 
                                     out_folder='test-data-sept5/',
                                     mask_shapefiles=test_shapefiles,
                                     n_shards=100)
            extractor.extract_data_over_patches(patches)
    if extract_train:
        for year in years:
            extractor = GEEExtractor(year, 
                                     out_gs_bucket=gs_bucket, 
                                     out_folder='train-data-sept10/', 
                                     mask_shapefiles=train_shapefiles,
                                     n_shards=30)
            for shapefile in train_shapefiles:
                if 'irrigated_train' in shapefile and 'unirrigated_train' not in shapefile:
                    extractor.extract_data_over_shapefile(shapefile, percent=10)
                elif 'wetlands' in shapefile:
                    extractor.extract_data_over_shapefile(shapefile, percent=2)
                else:
                    extractor.extract_data_over_shapefile(shapefile, percent=2)
    '''
