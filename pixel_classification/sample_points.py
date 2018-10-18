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

import fiona
from numpy import linspace, max, ceil
from numpy.random import shuffle
from pandas import DataFrame
from pyproj import Proj
from shapely.geometry import shape, Point, mapping

OPEN_WATER = [
    'MT_Wetlands_East_ow_1000_wgs84.shp',
    'WA_Wetlands_West_ow_1000_wgs84.shp',
    'CA_Wetlands_NorthCentral_ow_1000_wgs84.shp',
    'CA_Wetlands_SouthCentral_ow_1000_wgs84.shp',
    'CA_Wetlands_South_ow_1000_wgs84.shp',
    'WY_Wetlands_East_ow_1000_wgs84.shp',
    'OR_Wetlands_East_ow_1000_wgs84.shp',
    'NM_Wetlands_ow_1000_wgs84.shp',
    'CO_Wetlands_West_ow_1000_wgs84.shp',
    'ID_Wetlands_ow_1000_wgs84.shp',
    'AZ_Wetlands_ow_1000_wgs84.shp',
    'CO_Wetlands_East_ow_1000_wgs84.shp',
    'MT_Wetlands_West_ow_1000_wgs84.shp',
    'WA_Wetlands_East_ow_1000_wgs84.shp',
    'NV_Wetlands_South_ow_1000_wgs84.shp',
    'OR_Wetlands_West_ow_1000_wgs84.shp',
    'CA_Wetlands_North_ow_1000_wgs84.shp',
    'WY_Wetlands_West_ow_1000_wgs84.shp',
    'UT_Wetlands_ow_1000_wgs84.shp',
    'NV_Wetlands_North_ow_1000_wgs84.shp'
]

WETLAND = ['MT_Wetlands_East_wl_5000_wgs84.shp',
           'WA_Wetlands_West_wl_5000_wgs84.shp',
           'CA_Wetlands_NorthCentral_wl_5000_wgs84.shp',
           'CA_Wetlands_SouthCentral_wl_5000_wgs84.shp',
           'CA_Wetlands_South_wl_5000_wgs84.shp',
           'WY_Wetlands_East_wl_5000_wgs84.shp',
           'OR_Wetlands_East_wl_5000_wgs84.shp',
           'NM_Wetlands_wl_5000_wgs84.shp',
           'CO_Wetlands_West_wl_5000_wgs84.shp',
           'ID_Wetlands_wl_5000_wgs84.shp',
           'AZ_Wetlands_wl_5000_wgs84.shp',
           'CO_Wetlands_East_wl_5000_wgs84.shp',
           'MT_Wetlands_West_wl_5000_wgs84.shp',
           'WA_Wetlands_East_wl_5000_wgs84.shp',
           'NV_Wetlands_South_wl_5000_wgs84.shp',
           'OR_Wetlands_West_wl_5000_wgs84.shp',
           'CA_Wetlands_North_wl_5000_wgs84.shp',
           'WY_Wetlands_West_wl_5000_wgs84.shp',
           'UT_Wetlands_wl_5000_wgs84.shp',
           'NV_Wetlands_North_wl_5000_wgs84.shp']

IRR = {'ID': [('ID_2008_ESPA_WGS84_irr.shp', 2008),
              ('ID_1996_ESPA_WGS84_irr.shp', 1996),
              ('ID_2010_ESPA_WGS84_irr.shp', 2010),
              ('ID_2011_ESPA_WGS84_irr.shp', 2011),
              ('ID_1986_ESPA_WGS84_irr.shp', 1986),
              ('ID_2009_ESPA_WGS84_irr.shp', 2009),
              ('ID_2002_ESPA_WGS84_irr.shp', 2002),
              ('ID_2006_ESPA_WGS84_irr.shp', 2006)],

       'MT': [('MT_Irr_2018-2013_WGS84.shp', 2008),
              ('MT_Irr_2018-2013_WGS84.shp', 2009),
              ('MT_Irr_2018-2013_WGS84.shp', 2010),
              ('MT_Irr_2018-2013_WGS84.shp', 2011),
              ('MT_Irr_2018-2013_WGS84.shp', 2012),
              ('MT_Irr_2018-2013_WGS84.shp', 2013)],

       'WA': [('WA_Irr_WGS84_2009.shp', 2009),
              ('WA_Irr_WGS84_2010.shp', 2010),
              ('WA_Irr_WGS84_2012.shp', 2012),
              ('WA_Irr_WGS84_2000.shp', 2000),
              ('WA_Irr_WGS84_2004.shp', 2004),
              ('WA_Irr_WGS84_2011.shp', 2011),
              ('WA_Irr_WGS84_2007.shp', 2007),
              ('WA_Irr_WGS84_2006.shp', 2006),
              ('WA_Irr_WGS84_2017.shp', 2017),
              ('WA_Irr_WGS84_2015.shp', 2015),
              ('WA_Irr_WGS84_2002.shp', 2002),
              ('WA_Irr_WGS84_2008.shp', 2008),
              ('WA_Irr_WGS84_2001.shp', 2001),
              ('WA_Irr_WGS84_1999.shp', 1999),
              ('WA_Irr_WGS84_1998.shp', 1998),
              ('WA_Irr_WGS84_2003.shp', 2003),
              ('WA_Irr_WGS84_2005.shp', 2005),
              ('WA_Irr_WGS84_2014.shp', 2014),
              ('WA_Irr_WGS84_1996.shp', 1996),
              ('WA_Irr_WGS84_2016.shp', 2016),
              ('WA_Irr_WGS84_2013.shp', 2013)]}

IRR = {'ID': [('ID_2008_ESPA_WGS84_irr.shp', 2008),
              ('ID_1996_ESPA_WGS84_irr.shp', 1996),
              ('ID_2010_ESPA_WGS84_irr.shp', 2010),
              ('ID_2011_ESPA_WGS84_irr.shp', 2011),
              ('ID_1986_ESPA_WGS84_irr.shp', 1986),
              ('ID_2009_ESPA_WGS84_irr.shp', 2009),
              ('ID_2002_ESPA_WGS84_irr.shp', 2002),
              ('ID_2006_ESPA_WGS84_irr.shp', 2006)],

       'MT': [('MT_Irr_2018-2013_WGS84.shp', 2008),
              ('MT_Irr_2018-2013_WGS84.shp', 2009),
              ('MT_Irr_2018-2013_WGS84.shp', 2010),
              ('MT_Irr_2018-2013_WGS84.shp', 2011),
              ('MT_Irr_2018-2013_WGS84.shp', 2012),
              ('MT_Irr_2018-2013_WGS84.shp', 2013)],

       'WA': [('WA_Irr_WGS84_2009.shp', 2009),
              ('WA_Irr_WGS84_2010.shp', 2010),
              ('WA_Irr_WGS84_2012.shp', 2012),
              ('WA_Irr_WGS84_2000.shp', 2000),
              ('WA_Irr_WGS84_2004.shp', 2004),
              ('WA_Irr_WGS84_2011.shp', 2011),
              ('WA_Irr_WGS84_2007.shp', 2007),
              ('WA_Irr_WGS84_2006.shp', 2006),
              ('WA_Irr_WGS84_2017.shp', 2017),
              ('WA_Irr_WGS84_2015.shp', 2015),
              ('WA_Irr_WGS84_2002.shp', 2002),
              ('WA_Irr_WGS84_2008.shp', 2008),
              ('WA_Irr_WGS84_2001.shp', 2001),
              ('WA_Irr_WGS84_1999.shp', 1999),
              ('WA_Irr_WGS84_1998.shp', 1998),
              ('WA_Irr_WGS84_2003.shp', 2003),
              ('WA_Irr_WGS84_2005.shp', 2005),
              ('WA_Irr_WGS84_2014.shp', 2014),
              ('WA_Irr_WGS84_1996.shp', 1996),
              ('WA_Irr_WGS84_2016.shp', 2016),
              ('WA_Irr_WGS84_2013.shp', 2013)]}

IRR_RELATIVE_AREA = {'ID': 0.33,
                     'MT': 0.33,
                     'WA': 0.33}

YEARS = [1986, 1996, 2002, 2006, 2008, 2009, 2010, 2011, 2013]


class PointsRunspec(object):

    def __init__(self, root, **kwargs):
        self.root = root
        self.features = []
        self.object_id = 0
        self.year = None
        self.aea = Proj(
            '+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs')
        self.meta = None
        self.extracted_points = DataFrame(columns=['FID', 'X', 'Y', 'POINT_TYPE', 'YEAR'])

        if 'surface_water' in kwargs.keys():
            self.surface_water(kwargs['surface_water'])
        if 'wetlands' in kwargs.keys():
            self.wetlands(kwargs['wetlands'])
        if 'unirrigated' in kwargs.keys():
            self.unirrigated(kwargs['unirrigated'])
        if 'forest' in kwargs.keys():
            self.forest(kwargs['forest'])
        if 'irrigated' in kwargs.keys():
            self.irrigated(kwargs['irrigated'])

    def surface_water(self, n):
        print('surface water: {}'.format(n))
        n /= len(YEARS)
        for yr in YEARS:
            self.year = yr
            _files = [os.path.join(self.root, 'open_water', x) for x in OPEN_WATER]
            counts = [(f, int(float(n) / len(_files))) for f in _files]
            self.shapefile_area_count(_files)
            for s, c in counts:
                print(self.extracted_points.shape)
                self.create_sample_points(c, s, code=1)

    def wetlands(self, n):
        print('wetlands: {}'.format(n))
        for yr in YEARS:
            n /= len(YEARS)
            self.year = yr
            _files = [os.path.join(self.root, 'wetlands', x) for x in OPEN_WATER]
            areas = self.shapefile_area_count(_files)
            total_area = sum([x[1] for x in areas])
            samples = [(s, (a * n / total_area), ct) for s, a, ct in areas]
            for s, n, ct in samples:
                self.create_sample_points(n, s, code=1)

    def unirrigated(self, n):
        pass

    def forest(self, n):
        pass

    def irrigated(self, n):
        irr_path = os.path.join(self.root, 'irrigated')
        for k, v in IRR.items():
            years = len(v)
            for shp, yr in v:
                self.year = yr
                required_points = int(ceil(n * IRR_RELATIVE_AREA[k] / years))
                shp_path = os.path.join(irr_path, k, shp)
                self.create_sample_points(required_points, shp_path, code=0)

    def relative_irrigated_area(self):
        irr_path = os.path.join(self.root, 'irrigated')
        for k, v in IRR.items():
            shapes = [os.path.join(irr_path, k, x[0]) for x in v]
            areas = self.shapefile_area_count(shapes)
            total = sum([x[1] for x in areas])
            print('{} is {}'.format(k, total / 1e6))

    def shapefile_area_count(self, shapes):
        a = 0
        totals = []
        for shp in shapes:
            ct = 0
            with fiona.open(shp, 'r') as src:
                if not self.meta:
                    self.meta = src.meta
                for feat in src:
                    l = feat['geometry']['coordinates'][0]
                    if any(isinstance(i, list) for i in l):
                        l = l[0]
                    lon, lat = zip(*l)
                    x, y = self.aea(lon, lat)
                    cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
                    a += shape(cop).area
                    ct += 1
            totals.append((shp, a, ct))

        return totals

    def create_sample_points(self, n, shp, code):

        instance_ct = 0

        polygons = self._get_polygons(shp)
        shuffle(polygons)
        positive_area = sum([x.area for x in polygons])
        for i, poly in enumerate(polygons):
            fractional_area = poly.area / positive_area
            required_points = max([1, fractional_area * n])
            x_range, y_range = self._random_points(poly.bounds, n)
            poly_pt_ct = 0
            for coord in zip(x_range, y_range):
                if Point(coord[0], coord[1]).within(poly):
                    self._add_entry(coord, val=code)
                    poly_pt_ct += 1
                    instance_ct += 1

                if poly_pt_ct >= required_points:
                    break

            if instance_ct > n:
                break

    def _random_points(self, coords, n):
        min_x, max_x = coords[0], coords[2]
        min_y, max_y = coords[1], coords[3]
        x_range = linspace(min_x, max_x, num=2 * n)
        y_range = linspace(min_y, max_y, num=2 * n)
        shuffle(x_range), shuffle(y_range)
        return x_range, y_range

    def _add_entry(self, coord, val=0):

        self.extracted_points = self.extracted_points.append({'FID': int(self.object_id),
                                                              'X': coord[0],
                                                              'Y': coord[1],
                                                              'POINT_TYPE': val,
                                                              'YEAR': int(self.year)},
                                                             ignore_index=True)
        self.object_id += 1

    def save_sample_points(self, path):

        points_schema = {
            'properties': dict([('FID', 'int:10'), ('POINT_TYPE', 'int:10'), ('YEAR', 'int:10')]),
            'geometry': 'Point'}
        crs = {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'}
        meta = {'driver': 'ESRI Shapefile', 'schema': points_schema, 'crs': crs}

        with fiona.open(path, 'w', **meta) as output:
            for index, row in self.extracted_points.iterrows():
                props = dict([('FID', row['FID']),
                              ('POINT_TYPE', row['POINT_TYPE']),
                              ('YEAR', row['YEAR'])])

                pt = Point(row['X'], row['Y'])
                output.write({'properties': props,
                              'geometry': mapping(pt)})
        return None

    def _get_polygons(self, vector):
        with fiona.open(vector, 'r') as src:
            polys = []
            bad_geo_count = 0
            for feat in src:
                try:
                    geo = shape(feat['geometry'])
                    polys.append(geo)
                except AttributeError:
                    bad_geo_count += 1

        return polys


if __name__ == '__main__':
    home = os.path.expanduser('~')
    gis = os.path.join(home, 'IrrigationGIS', 'EE_sample')
    extract = os.path.join(home, 'IrrigationGIS', 'EE_extracts')
    kwargs = {'irrigated': 1000,
              'surface_water': 500,
              'wetlands': 1000
              }
    prs = PointsRunspec(gis, **kwargs)
    prs.save_sample_points(os.path.join(extract, 'sample_100.shp'))

# ========================= EOF ====================================================================
