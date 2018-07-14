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
from pixel_classification.runspec import Idaho, Montana, Nevada, Oregon, Washington

home = os.path.expanduser('~')
ROOT = os.path.join(home, 'IrrigationGIS', 'western_states_irrgis')


OBJECT_MAP = {'ID': Idaho,
              'MT': Montana,
              'NV': Nevada,
              'OR': Oregon,
              'WA': Washington}


def build_compiled_feature_array():
    for key, obj in OBJECT_MAP.items():
        path = os.path.join(ROOT, key)
        geo = obj(path)



if __name__ == '__main__':
    build_compiled_feature_array()
# ========================= EOF ====================================================================
