import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from glob import glob
from numpy import sum as nsum

from losses import *
from models import *
from data_generators import StackDataGenerator
from train_utils import confusion_matrix_from_generator, timeseries_confusion_matrix_from_generator


if __name__ ==  '__main__':

    #import numpy as np
    #arr = np.array([[8.3913000e+04,1.8664000e+04,1.0272000e+04],
    #                 [1.6963000e+04,1.1221567e+07,7.2342400e+05],
    #                 [1.1794000e+04,1.7684930e+06,3.9401140e+06]])
    #n_classes = 3
    #precision_dict = {}
    #recall_dict = {}
    #for i in range(n_classes):
    #    precision_dict[i] = 0
    #    recall_dict[i] = 0
    #for i in range(n_classes):
    #    precision_dict[i] = arr[i, i] / np.sum(arr[i, :]) # row i
    #    recall_dict[i] = arr[i, i] / np.sum(arr[:, i]) # column i
    #print(precision_dict)
    #print(recall_dict)
#start idx 0
#[  656456. 62292313. 38808916.]
#[[1.6025900e+05 4.2173400e+05 7.4463000e+04]
# [1.1990000e+05 5.7056893e+07 5.1155200e+06]
# [3.6746000e+04 6.3206270e+06 3.2451543e+07]]
#
# p:{0: 0.5057004465060507, 1: 0.8943191247972899, 2: 0.8621208130616171}
# r:{0: 0.24412755767332464, 1: 0.9159539958004128, 2: 0.836187823437274}
#[[  269559.   320481.    66416.]
# [  627005. 51898532.  9766776.]
# [  116948.  6658370. 32033598.]]
#---------------
#start idx 1
#[  656456. 62292313. 38808916.]
#[[  269559.   320481.    66416.]
# [  627005. 51898532.  9766776.]
# [  116948.  6658370. 32033598.]]
#
# p:{0: 0.26596527717481394, 1: 0.8814680503037983, 2: 0.7651314562210286}
# r:{0: 0.4106276734465067, 1: 0.8331450463237735, 2: 0.8254185198061188}
#[[2.5838800e+05 3.7004600e+05 2.8022000e+04]
# [1.3530700e+05 5.6543391e+07 5.6136150e+06]
# [2.5605200e+05 1.0854594e+07 2.7698270e+07]]
#---------------
#start idx 2
#[  656456. 62292313. 38808916.]
#[[2.5838800e+05 3.7004600e+05 2.8022000e+04]
# [1.3530700e+05 5.6543391e+07 5.6136150e+06]
# [2.5605200e+05 1.0854594e+07 2.7698270e+07]]
#
# p:{0: 0.3976747872633502, 1: 0.8343667384994556, 2: 0.8307842610358811}
# r:{0: 0.39361053901556237, 1: 0.9077105709656342, 2: 0.7137089322463941}
#[[3.1228000e+05 3.1129800e+05 3.2878000e+04]
# [3.4318400e+05 5.8021595e+07 3.9275340e+06]
# [2.2140100e+05 9.8592890e+06 2.8728226e+07]]
#---------------
#start idx 3
#[  656456. 62292313. 38808916.]
#[[3.1228000e+05 3.1129800e+05 3.2878000e+04]
# [3.4318400e+05 5.8021595e+07 3.9275340e+06]
# [2.2140100e+05 9.8592890e+06 2.8728226e+07]]
#
# p:{0: 0.3561323578886145, 1: 0.8508540612470796, 2: 0.8788443862359759}
# r:{0: 0.47570591174427534, 1: 0.9314406899612157, 2: 0.740248091443729}
#[[2.8896800e+05 3.2847300e+05 3.9015000e+04]
# [7.5190000e+04 5.4773090e+07 7.4440330e+06]
# [1.2251700e+05 4.7295680e+06 3.3956831e+07]]
#---------------
#start idx 4
#[  656456. 62292313. 38808916.]
#[[2.8896800e+05 3.2847300e+05 3.9015000e+04]
# [7.5190000e+04 5.4773090e+07 7.4440330e+06]
# [1.2251700e+05 4.7295680e+06 3.3956831e+07]]
#
# p:{0: 0.5937596958956183, 1: 0.9154613841413093, 2: 0.8194239901134847}
# r:{0: 0.440194011479825, 1: 0.8792913180154347, 2: 0.8749749928598882}
#[[2.6443100e+05 3.7678800e+05 1.5237000e+04]
# [1.3527100e+05 5.8336752e+07 3.8202900e+06]
# [1.2181400e+05 7.7686780e+06 3.0918424e+07]]
#---------------
#start idx 5
#[  656456. 62292313. 38808916.]
#[[2.6443100e+05 3.7678800e+05 1.5237000e+04]
# [1.3527100e+05 5.8336752e+07 3.8202900e+06]
# [1.2181400e+05 7.7686780e+06 3.0918424e+07]]
#
# p:{0: 0.5070429286925041, 1: 0.8774790275498932, 2: 0.8896376702608575}
# r:{0: 0.40281603032038704, 1: 0.9365000140547036, 2: 0.7966835249920405}
#[[3.0331100e+05 3.2977100e+05 2.3374000e+04]
# [1.0975500e+05 5.8753091e+07 3.4294670e+06]
# [1.2158100e+05 6.0948860e+06 3.2592449e+07]]
#---------------
#start idx 6
#[  656456. 62292313. 38808916.]
#[[3.0331100e+05 3.2977100e+05 2.3374000e+04]
# [1.0975500e+05 5.8753091e+07 3.4294670e+06]
# [1.2158100e+05 6.0948860e+06 3.2592449e+07]]
#
# p:{0: 0.5673107676653941, 1: 0.9014286747065885, 2: 0.9042082613290113}
# r:{0: 0.4620431529302802, 1: 0.9431836477158907, 2: 0.8398185870483988}
#[[2.9467500e+05 2.6421100e+05 9.7570000e+04]
# [1.8055200e+05 5.5271868e+07 6.8398930e+06]
# [2.7967000e+04 6.3804290e+06 3.2400520e+07]]
#---------------
#start idx 7
#[  656456. 62292313. 38808916.]
#[[2.9467500e+05 2.6421100e+05 9.7570000e+04]
# [1.8055200e+05 5.5271868e+07 6.8398930e+06]
# [2.7967000e+04 6.3804290e+06 3.2400520e+07]]
#
# p:{0: 0.5856091288846846, 1: 0.8926838703500527, 2: 0.8236446693263353}
# r:{0: 0.4488876634534531, 1: 0.8872983733964093, 2: 0.8348730997794425}
#[[2.2450500e+05 2.6122500e+05 1.7072600e+05]
# [8.0536000e+04 5.5395186e+07 6.8165910e+06]
# [4.5437000e+04 7.8964020e+06 3.0867077e+07]]
#---------------
#start idx 8
#[  656456. 62292313. 38808916.]
#[[2.2450500e+05 2.6122500e+05 1.7072600e+05]
# [8.0536000e+04 5.5395186e+07 6.8165910e+06]
# [4.5437000e+04 7.8964020e+06 3.0867077e+07]]
#
# p:{0: 0.6405680242411792, 1: 0.8716401900258923, 2: 0.8154159593731708}
# r:{0: 0.3419955031258759, 1: 0.8892780398120712, 2: 0.7953604527372009}

    string = '2015_34_27test-data, 2015_39_26test-data, 2015_35_27test-data, 2015_39_27test-data,\
    2015_36_26test-data, 2015_39_28test-data, 2015_36_27test-data, 2015_40_27test-data,\
    2015_36_28test-data, 2015_40_28test-data, 2015_37_26test-data, 2015_41_27test-data,\
    2015_37_28test-data, 2015_41_28test-data, 2015_38_26test-data, 2015_42_27test-data,\
    2015_38_27test-data, 2015_38_28test-data'

    string = string.replace(",", " ")
    string = string.split(" ")
    string = [s for s in string if len(s)]

    model = unet((None, None, 98), n_classes=3, initial_exp=5)
    ### BEST MODEL: but precision is 89.2 and recall is 0.94
    model_path = './model_0.969-0.910.h5'
    ### BEST MODEL k
    model.load_weights(model_path)
    batch_size = 16
    min_images = 14

    total_cmat = np.zeros((3, 3))
    for test_data_path in string:
        if '41_28' not in test_data_path:
            continue
        n_classes = 3
        test_data_path = os.path.join("/media/synology/", test_data_path)

        final_cmat = np.zeros((3, 3))
        for start_idx in range(0, 12):
            test_generator = StackDataGenerator(data_directory=test_data_path, batch_size=batch_size,
                    training=False, min_images=min_images, start_idx=start_idx)
            try:
                cmat, prec, recall = confusion_matrix_from_generator(test_generator, batch_size, 
                        model, n_classes=n_classes, time_dependent=False,
                        print_mat=False)
            except ValueError as e:
                break
            print('--------- start idx {} -----------'.format(start_idx))
            print(cmat)
            print(prec)
            print(recall)
