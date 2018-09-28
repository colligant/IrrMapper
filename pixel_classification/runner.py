# ===============================================================================
# Copyright 2018 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import os
import sys

abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)
from numpy import vstack

from pixel_classification.runspec import Montana, Nevada, Oregon, Utah, Washington
from pixel_classification.prepare_images import ImageStack
from pixel_classification.compose_array import PixelTrainingArray as Pta
from pixel_classification.tf_multilayer_perceptron import mlp
from pixel_classification.classify import classify_multiproc
from pixel_classification.crop_data_layer import CropDataLayer



OBJECT_MAP = {
    # 'MT': Montana,
    'NV': Nevada,
    'OR': Oregon,
    'UT': Utah,
    'WA': Washington
}


def build_training_feature_array(project_root, training_root, sat=8):

    for key, obj in OBJECT_MAP.items():

        project_state_dir = os.path.join(project_root, key)

        if not os.path.isdir(project_state_dir):
            os.mkdir(project_state_dir)

        geography = os.path.join(training_root, key)
        geo = obj(geography)

        i = None
        if geo.sat == sat:
            i = ImageStack(root=project_state_dir, satellite=geo.sat, path=geo.path, row=geo.row,
                           n_landsat=3, year=geo.year, max_cloud_pct=70)
            i.build_all()
            p = Pta(root=i.root, geography=geo, instances=5000, overwrite_array=True,
                    overwrite_points=False, ancillary_rasters=i.ancillary_rasters)
            p.extract_sample(save_points=True)

        i.warp_vrt()


def build_model(project_root, path, model_path):
    first = True
    for key, obj in OBJECT_MAP.items():
        root = os.path.join(project_root, key)
        geo = obj(root)
        pkl_data = Pta(root=root, geography=geo, pkl_path=os.path.join(root, 'data.pkl'))
        if first:
            training_data = {'data': pkl_data.data, 'target_values': pkl_data.target_values,
                             'features': pkl_data.features, 'model_map': pkl_data.model_map,
                             }
            first = False
        else:
            training_data = concatenate_training_data(training_data, pkl_data)

        print('Shape {}: {}'.format(key, pkl_data.data.shape))

    p = Pta(from_dict=training_data)
    p.to_pickle(training_data, path)
    model_path = mlp(p, model_path)

    return model_path


def concatenate_training_data(existing, new_data):
    existing_array = existing['data']
    add_array = new_data.data
    new_array = vstack((existing_array, add_array))

    existing_features = existing['features']
    add_features = new_data.features
    new_features = vstack((existing_features.reshape((existing_features.shape[0], 1)),
                           add_features.reshape((add_features.shape[0], 1))))

    existing_labels = existing['target_values']
    add_labels = new_data.target_values
    new_lables = vstack((existing_labels.reshape((existing_labels.shape[0], 1)),
                         add_labels.reshape((add_labels.shape[0], 1))))

    existing['model_map'].update(new_data.model_map)

    concatenated = {'data': new_array, 'target_values': new_lables, 'features': new_features,
                    'model_map': existing['model_map']}

    return concatenated


if __name__ == '__main__':
    home = os.path.expanduser('~')

    training = os.path.join(home, 'IrrigationGIS', 'western_states_irrgis')
    model_data = os.path.join(abspath, 'model_data')
    project = os.path.join(model_data, 'allstates_3')

    # if not os.path.isdir(project):
    #     os.mkdir(project)
    #
    # build_training_feature_array(project_root=project, training_root=training)

    data_path = os.path.join(project, 'data.pkl')
    model = os.path.join(project, 'model.ckpt')
    # build_model(project, data_path, model)

    for key, val in OBJECT_MAP.items():
        geo_folder = os.path.join(project, key)
        save_array = os.path.join(geo_folder, 'array.npy')
        geo_data = os.path.join(geo_folder, 'data.pkl')
        cdl_path = os.path.join(geo_folder, 'cdl_mask.tif')
        classify_multiproc(model, geo_data, array_outfile=save_array, mask=cdl_path)

# ========================= EOF ====================================================================
