import ee
# ee.Authenticate()
ee.Initialize()
import tensorflow as tf
import time
import os
import numpy as np
from random import shuffle

import ee_utils
from shapefile_spec import shape_to_year_and_count as SHP_TO_YEAR_AND_COUNT

KERNEL_SIZE = 64
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
list_ = ee.List.repeat(1, KERNEL_SIZE)
lists = ee.List.repeat(list_, KERNEL_SIZE)
KERNEL = ee.Kernel.fixed(KERNEL_SIZE, KERNEL_SIZE, lists)
GS_BUCKET = 'ee-irrigation-mapping'

class GEEExtractor:

    def __init__(self, year, out_folder, mask_shapefiles, n_shards=10):

        self.year = year
        self.out_folder = out_folder
        self.n_shards = n_shards
        self.mask_shapefiles = mask_shapefiles

        self._construct_data_stack()

    
    def _construct_data_stack(self):
        image_stack = ee_utils.preprocess_data(self.year).toBands()
        features = [feat['id'] for feat in image_stack.getInfo()['bands']]
        self.features = features + ['constant']
        self.shapefile_to_feature_collection = \
                ee_utils.temporally_filter_features(self.mask_shapefiles, self.year)
        class_labels = ee_utils.create_class_labels(self.shapefile_to_feature_collection)
        self.image_stack = ee.Image.cat([image_stack, class_labels]).float()
        self.data_stack = self.image_stack.neighborhoodToArray(KERNEL)

    def extract_data_over_centroids(mask_shapefiles):
        for shapefile, feature_collection in self.shapefile_to_feature_collection.items():
            centroids = feature_collection.map(lambda feature: \
                    ee.Feature(feature.geometry().centroid()))
            centroids = centroids.toList(centroids.size())
            n_features = SHP_TO_YEAR_AND_COUNT[os.path.basename(shapefile)][self.year]
            out_filename = self._create_filename(shapefile)
            geometry_sample = ee.ImageCollection([])
            indices = [int(i) for i in np.random.choice(n_features, size=n)]
            feature_count = 0
            for idx in indices:
                sample = data_stack.sample(
                        region=ee.Feature(centroids.get(idx)).geometry(),
                        scale=30,
                        numPixels=1,
                        tileScale=8
                        )
                geometry_sample = geometry_sample.merge(sample)
                if (feature_count+1) % n_shards == 0:
                    geometry_sample = self._create_and_start_table_task(geometry_sample)
                feature_count += 1
            # take care of leftovers
            self._create_and_start_table_task(geometry_sample)


    def extract_data_over_patches(self, patch_shapefile):
        if isinstance(patch_shapefile, list):
            for p_shp in patch_shapefile:
                p_shp = ee.FeatureCollection(p_shp)
                patches = p_shp.toList(centroids.size())
                out_filename = self._create_filename(patch_shapefile)
                for idx in range(patches.size().getInfo()):
                    patch = ee.Feature(patches.get(idx))
                    self._create_and_start_image_task(patch, out_filename)
        else:
            patches = ee.FeatureCollection(patch_shapefile)
            patches = patches.toList(patches.size())
            out_filename = self._create_filename(patch_shapefile)
            for idx in range(patches.size().getInfo()):
                patch = ee.Feature(patches.get(idx))
                self._create_and_start_image_task(patch, out_filename)
                break


    def extract_data_over_shapefiles(self):
        for shapefile, feature_collection in self.shapefile_to_feature_collection.items():
            polygons = polygons.toList(centroids.size())
            n_features = SHP_TO_YEAR_AND_COUNT[os.path.basename(shapefile)][self.year]
            out_filename = self._create_filename(shapefile)
            print(out_filename)
            geometry_sample = ee.ImageCollection([])
            indices = [int(i) for i in np.random.choice(n_features, size=n)]
            feature_count = 0
            for idx in indices:
                sample = self.data_stack.sample(
                        region=ee.Feature(polygons.get(idx)).geometry(),
                        scale=30,
                        numPixels=1,
                        tileScale=8
                        )
                geometry_sample = geometry_sample.merge(sample)
                if (feature_count+1) % n_shards == 0:
                    geometry_sample = self._create_and_start_table_task(geometry_sample,
                            out_filename)
                feature_count += 1
            # take care of leftovers
            self._create_and_start_table_task(geometry_sample)

    def _create_and_start_image_task(self, patch, out_filename):
        task = ee.batch.Export.image.toCloudStorage(
                image=self.image_stack,
                bucket=GS_BUCKET,
                description=out_filename + str(time.time()),
                fileNamePrefix=self.out_folder +  out_filename + str(time.time()),
                fileFormat='TFRecord',
                region=patch.geometry(),
                scale=30,
                formatOptions={'patchDimensions':256,
                               'compressed':True},
                )
        self._start_task_and_handle_exception(task)
    
    def _create_and_start_table_task(self, geometry_sample, out_filename):
        task = ee.batch.Export.table.toCloudStorage(
                collection=geometry_sample,
                description=out_filename + str(time.time()),
                bucket=GS_BUCKET,
                fileNamePrefix=self.out_folder + out_filename + str(time.time()),
                fileFormat='TFRecord',
                selectors=features
                )
        geom = self._start_task_and_handle_exception(task)
        return geom

    def _start_task_and_handle_exception(self, task):

        try:
            task.start()
        except ee.ee_exception.EEException:
            print('waiting to export, sleeping for 50 minutes')
            time.sleep(3000)
            task.start()
        return ee.ImageCollection([])

    def _create_filename(self, shapefile):

        return os.path.basename(shapefile) + str(self.year)


''' steps:

1. Make grid from extent.
2. Split grid into test/train.
3. Clip mask shapefiles to grid.
4. Get intersections of grid shapefiles and mask shapefiles
5. Upload to earth-engine.
6. Extract data.

'''


if __name__ == '__main__':

    root = 'users/tcolligan0/train-aug12/'
    train_shapefiles = ['irrigated_train', 'uncultivated_train', 'unirrigated_train',
            'wetlands_train', 'fallow_train']
    train_shapefiles = [root + s for s in train_shapefiles]

    patches = 'users/tcolligan0/test_block_aug14'
    extractor = GEEExtractor(2013, 'aaa256/', train_shapefiles)
    extractor.extract_data_over_patches(patches)
