import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

import utils
import config


test = utils.make_test_dataset(os.path.join('/tmp/', 
    config.TEST_BASE), batch_size=2*config.BATCH_SIZE, add_ndvi=True)

class_counts = np.zeros((5,))

for features, labels in train:
    masked_true, _ = utils.mask_unlabeled_values(labels, labels)
    print(np.unique(masked_true.numpy(), return_counts=True))
