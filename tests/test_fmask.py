# ===============================================================================
# Copyright 2017 dgketchum
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
import unittest
from numpy import count_nonzero

from sat_image.image import Landsat5, Landsat7, Landsat8
from sat_image.fmask import Fmask


class FmaskTestCaseL5(unittest.TestCase):
    def setUp(self):
        self.dirname_cloud = 'satellite_image/tests/data/fmask_test/lt5_fmask'
        self.image = Landsat5(self.dirname_cloud)

    def test_instantiate_fmask(self):
        self.assertIsInstance(self.image, Landsat5)

    def test_get_potential_cloud_layer(self):
        f = Fmask(self.image)
        self.assertIsInstance(f, Fmask)
        cloud, shadow, water = f.cloud_mask()
        c_ct, s_ct = count_nonzero(cloud), count_nonzero(shadow)
        w_ct = count_nonzero(water)
        self.assertEqual(c_ct, 229040)
        self.assertEqual(s_ct, 37570)
        self.assertEqual(w_ct, 9423)


class FmaskTestCaseL7(unittest.TestCase):
    def setUp(self):
        self.dirname_cloud = 'satellite_image/tests/data/fmask_test/le7_fmask'
        self.image = Landsat7(self.dirname_cloud)

    def test_instantiate_fmask(self):
        self.assertIsInstance(self.image, Landsat7)

    def test_get_potential_cloud_layer(self):
        ndvi = self.image.ndvi()
        f = Fmask(self.image)
        self.assertIsInstance(f, Fmask)
        cloud, shadow, water = f.cloud_mask()
        combo = f.cloud_mask(combined=True)
        c_ct, s_ct = count_nonzero(cloud), count_nonzero(shadow)
        w_ct = count_nonzero(water)
        self.assertEqual(c_ct, 133502)
        self.assertEqual(s_ct, 21960)
        self.assertEqual(w_ct, 29678)
        home = os.path.expanduser('~')
        outdir = os.path.join(home, 'images', 'sandbox')
        f.save_array(ndvi, os.path.join(outdir, 'ndvi.tif'))


class FmaskTestCaseL8(unittest.TestCase):
    def setUp(self):
        self.dirname_cloud = 'satellite_image/tests/data/fmask_test/lc8_fmask'
        self.image = Landsat8(self.dirname_cloud)

    def test_instantiate_fmask(self):
        self.assertIsInstance(self.image, Landsat8)

    def test_get_potential_cloud_layer(self):
        # cloud and shadow have been visually inspected
        # test have-not-changed
        f = Fmask(self.image)
        self.assertIsInstance(f, Fmask)
        cloud, shadow, water = f.cloud_mask()
        combo = f.cloud_mask(combined=True)
        c_ct, s_ct = count_nonzero(cloud), count_nonzero(shadow)
        w_ct, combo_ct = count_nonzero(water), count_nonzero(combo)
        self.assertEqual(c_ct, 145184)
        self.assertEqual(s_ct, 41854)
        self.assertEqual(w_ct, 87399)
        self.assertEqual(combo_ct, 200182)


if __name__ == '__main__':
    unittest.main()

# ===============================================================================
# home = os.path.expanduser('~')
# outdir = os.path.join(home, 'images', 'sandbox')
# f.save_array(cloud, os.path.join(outdir, 'cloud_mask_l8.tif'))
# f.save_array(shadow, os.path.join(outdir, 'shadow_mask_l8.tif'))
# f.save_array(water, os.path.join(outdir, 'water_mask_l8.tif'))
