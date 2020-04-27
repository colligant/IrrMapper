import numpy as np
import pdb
import os
import matplotlib.pyplot as plt

from rasterio import open as rasopen
from collections import defaultdict
from random import sample, shuffle
from tensorflow.keras.utils import Sequence
from glob import glob


def _to_categorical(a, nodata=-9999, n_classes=3):
    # channels first
    a = np.squeeze(a)
    one_hot = np.zeros((n_classes, a.shape[0], a.shape[1]))
    for i in range(n_classes):
        one_hot[i][a == i] = 1 
    return one_hot


def _load_image(f, image=False):
    with rasopen(f, 'r') as src:
        im = src.read()
    return im


class StackDataGenerator(Sequence): 


    def __init__(self, data_directory, batch_size, 
            image_suffix='*.tif', training=True, only_irrigated=False,
            random_start_date=False, steps_per_epoch=None,
            min_images=None, secondary_data_directory=None,
            start_idx=None):

        self.classes = [d for d in os.listdir(os.path.join(data_directory, 'images')) if \
                os.path.isdir(os.path.join(data_directory, 'images', d))]
        if only_irrigated:
            self.classes = [c for c in self.classes if 'irrigated' in c]
            self.classes = [c for c in self.classes if 'unirrigated' not in c]
        if len(self.classes) == 0:
            raise ValueError('no directories in data directory {}'.format(data_directory))

        self.random_start_date = random_start_date
        self.secondary_data_directory = secondary_data_directory
        self.steps_per_epoch = steps_per_epoch
        self.start_idx = start_idx
        self.data_directory = data_directory
        self.training = training
        self.min_images = min_images
        self.batch_size = batch_size
        self.image_suffix = image_suffix
        self.n_classes = len(self.classes)
        self.min_instances = np.inf
        self.n_instances = 0
        self.index_to_class = {}
        self._create_file_dictionaries(data_directory)
        self._create_file_list()
    

    def _create_file_dictionaries(self, data_directory):
        self.class_to_image_files = {}
        self.class_to_mask_files = {}
        self.class_to_n_instances = {}
        for idx, d in enumerate(self.classes):
            self.index_to_class[idx] = d
            image_files = glob(os.path.join(data_directory, 'images', d, self.image_suffix))
            if self.secondary_data_directory is not None:
                image_files.extend(glob(os.path.join(self.secondary_data_directory, 'images', d,
                    self.image_suffix)))
            if not len(image_files):
                print("no training data for {} class".format(d))
                self.classes.remove(d)
                continue
            mask_files = [s.replace('images', 'masks') for s in image_files]

            self.class_to_image_files[d] = image_files
            self.class_to_mask_files[d] = mask_files

            if len(image_files) < self.min_instances:
                self.min_instances = len(image_files)


    def _create_file_list(self):
        self.images = []
        self.masks = []
        if self.training:
            # balance the number of examples per epoch
            for idx, d in enumerate(self.classes):
                l = len(self.class_to_image_files[d])
                indices = np.random.choice(np.arange(l), 
                        size=self.min_instances, replace=False)
                self.images.extend(list(np.asarray(self.class_to_image_files[d])[indices]))
                self.masks.extend(list(np.asarray(self.class_to_mask_files[d])[indices]))
            # shuffle again
            indices = np.random.choice(np.arange(len(self.images)), len(self.images), replace=False)
            self.images = list(np.asarray(self.images)[indices])
            self.masks = list(np.asarray(self.masks)[indices])
            if self.steps_per_epoch is not None:
                self.images = self.images[:self.steps_per_epoch*self.batch_size]
                self.masks = self.masks[:self.steps_per_epoch*self.batch_size]
            self.n_instances = len(self.images)
        else:
            # just grab the whole dataset if not training for consistent 
            # test metrics
            for idx, d in enumerate(self.classes):
                self.images.extend(self.class_to_image_files[d])
                self.masks.extend(self.class_to_mask_files[d])
            self.n_instances = len(self.images)


    def _do_nothing(self, image, labels):
        return image, labels

    def _hflip(self, image, labels):
        return np.fliplr(image), np.fliplr(labels)

    def augment(self, image, labels):
        return np.random.choice([self._do_nothing, self._hflip])(image, labels)

    def _conform_channels(self, image, min_images, 
            random_start_date=True):
        '''
        LANDSAT image stacks may have different number of bands.
        This function makes them all have the same number of bands 
        according to min_rgb_images.
        '''
        n_bands = 7

        n_rgb = image.shape[0] // n_bands
        if self.start_idx is not None:
            #idx = np.arange(4*n_bands, 19*n_bands)
            #good = idx < 5*n_bands
            #good = good | (idx >= 6*n_bands)
            #idx = idx[good]
            return image[self.start_idx*n_bands:self.start_idx*n_bands + 14*n_bands]
        if n_rgb > min_images:
            # just cut off end? or beginning?
            diff = n_rgb - min_images
            if not self.training:
                return image[:-diff*n_bands]
            start_idx = np.random.randint(diff)
            if np.random.randint(2) and random_start_date:
                return image[start_idx*n_bands:start_idx*n_bands + min_images*n_bands]
            else:
                return image[:-diff*n_bands]
        else:
            return image

    def __getitem__(self, idx):
        # since I'm just dealing with RGB, split into sequences of
        # size (batch, timesteps, height, width, depth)
        masks = self.masks[self.batch_size*idx:self.batch_size*(idx+1)]
        images = self.images[self.batch_size*idx:self.batch_size*(idx+1)]
        images = [_load_image(f) for f in images]
        images = [self._conform_channels(image, self.min_images, 
            self.random_start_date) for image in images]
        masks = [_to_categorical(_load_image(f)) for f in masks]
        images = np.swapaxes(np.asarray(images), 1, 3).astype(np.float)
        masks = np.swapaxes(np.asarray(masks), 1, 3).astype(np.float)
        image_sequence = np.asarray(images)
        mask_sequence = np.asarray(masks)
        return image_sequence, mask_sequence


    def on_epoch_end(self):
        if self.training:
            self._create_file_list()
        else:
            # do nothing...
            pass


    def __len__(self):
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch

        return int(np.ceil(self.n_instances // self.batch_size))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sys import stdout
    from shutil import copyfile
    oss = os.path.splitext
    osb = os.path.dirname
    train_path = '/data/thomas-training-data/2015-test-all-ims/train/'
    dg = StackDataGenerator(train_path, 32, training=True,
            min_images=9, only_irrigated=False)
    removed = 0
    root = '/media/synology/train/'
    os.makedirs(root, exist_ok=True)
#    for i, m in zip(dg.images, dg.masks):
#        idst = i.replace(train_path, root)
#        mdst = m.replace(train_path, root)
#        print(i, m)
#        os.makedirs(osb(oss(idst)[0]), exist_ok=True)
#        os.makedirs(osb(oss(mdst)[0]), exist_ok=True)
#        copyfile(i, idst)
#        copyfile(m, mdst)
