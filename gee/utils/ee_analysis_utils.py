import ee
import os
ee.Initialize()


ACRES_PER_SQUARE_METER = 0.000247105

def irrigated_acres_by_crop(irrigated_raster, county_shapefile, year):
    if county_shapefile is None:
        region = ee.FeatureCollection('users/tcolligan0/Montana')
    else:
        region = ee.FeatureCollection(county_shapefile)

    irr = ee.Image(irrigated_raster).select(["b1", "b2", "b3"], ["irr", "unirr", "uncult"])

    # do a homebrew argmax
    mask = irr.select("irr").gt(irr.select("unirr"));
    mask = mask.multiply(irr.select("irr").gt(irr.select("uncult")))
    mask = mask.updateMask(mask.gt(0.5).toFloat())

    # pixelArea is an ee builtin
    area_raster = ee.Image.pixelArea().multiply(mask);

    def clip_to_region(feature):
        name = feature.get("NAME")
        return area_raster.clip(feature).set("NAME", name)

    irr_ic = ee.ImageCollection(region.map(clip_to_region))
    cdl = ee.Image('USDA/NASS/CDL/{}'.format(year))
    cdl = cdl.updateMask(mask).select('cropland')

    region = region.toList(region.size())
    areas = cdl.reduceRegion(
              reducer=ee.Reducer.frequencyHistogram(),
              geometry=ee.Feature(region.get(0)).geometry(),
              crs="EPSG:5070",
              scale=30,
              maxPixels=1e10)
    return areas.getInfo()

def irrigated_predictions_by_county(irrigated_raster, county_shapefile, year):
    counties = ee.FeatureCollection(county_shapefile)
    irr = ee.Image(irrigated_raster).select(["b1", "b2", "b3"], ["irr", "unirr", "uncult"])

    # do a homebrew argmax
    mask = irr.select("irr").gt(irr.select("unirr"));
    mask = mask.multiply(irr.select("irr").gt(irr.select("uncult")))

    # pixelArea is an ee builtin
    area_raster = ee.Image.pixelArea().multiply(mask);

    def clip_to_counties(feature): 
        name = feature.get("NAME")
        return area_raster.clip(feature).set("NAME", name)

    irr_ic = ee.ImageCollection(counties.map(clip_to_counties))

    def reduce_area(image):
        areas = image.reduceRegion(
              reducer=ee.Reducer.sum(),
              geometry=image.geometry(),
              crs="EPSG:5070",
              scale=30,
              maxPixels=1e9
              )

        feature = ee.Feature(None, {'AREA': areas.get('area'),
            'NAME': image.get('NAME'), 'YEAR': year}) # None for a null geometry
        return feature

    areas = irr_ic.map(reduce_area)

    return areas

def timeseries_met_data_mt(year, start_date='{}-01-01', end_date='{}-12-31'):

    region = ee.FeatureCollection('users/tcolligan0/Montana')
    region = region.toList(region.size())
    gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
    gridmet = gridmet.filter(ee.Filter.date(start_date.format(year), end_date.format(year))).sum()
    gridmet = gridmet.clip(ee.Feature(region.get(0)).geometry())
    gridmet = gridmet.select('pr')

    sums = gridmet.reduceRegion(
              reducer=ee.Reducer.sum(),
              geometry=ee.Feature(region.get(0)).geometry(),
              crs="EPSG:5070",
              scale=30,
              maxPixels=1e10)
    print(sums.getInfo())

def export_timeseries_of_met_data(county_shapefile, year, gridmet_variable='pr', 
        start_date='{}-01-01', end_date='{}-12-31', reduction_type='sum'):
    counties = ee.FeatureCollection(county_shapefile)
    gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET")
    gridmet = gridmet.filter(ee.Filter.date(start_date.format(year), end_date.format(year))).sum()
    gridmet = gridmet.select(gridmet_variable)

    if reduction_type.lower() == 'sum':
        gridmet = gridmet.sum()
    elif reduction_type.lower() == 'mean':
        gridmet = gridmet.mean()
    else: raise ValueError('Only sum, mean supported for reduction_type')
    
    def clip_to_counties(feature): 
        name = feature.get("NAME")
        return gridmet.clip(feature).set("NAME", name)

    gridmet_ic = ee.ImageCollection(counties.map(clip_to_counties))

    def reduce_(image):

        if reduction_type == 'sum':
            reduction = image.reduceRegion(
                  reducer=ee.Reducer.sum(),
                  geometry=image.geometry(),
                  crs="EPSG:5070",
                  scale=30,
                  maxPixels=1e9
                  )
        else:
            reduction = image.reduceRegion(
                  reducer=ee.Reducer.mean(),
                  geometry=image.geometry(),
                  crs="EPSG:5070",
                  scale=30,
                  maxPixels=1e9
                  )

        feature = ee.Feature(None, {'{}_{}'.format(reduction_type.upper(),
            gridmet_variable.upper()): reduction.get(gridmet_variable), 
            'NAME': image.get('NAME'), 'YEAR': year}) 
        
        return feature
    
    sum_met_variables = gridmet_ic.map(reduce_)
    return sum_met_variables


def mask_irrigated_raster_to_cdl(irrigated_raster, cdl_class, county_shapefile, year):

    counties = ee.FeatureCollection(county_shapefile)
    cdl = ee.Image('USDA/NASS/CDL/{}'.format(year))
    cdl = cdl.select('cropland');
    irr = ee.Image(irrigated_raster).select(["b1", "b2", "b3"], ["irr", "unirr", "uncult"])

    dataset = ee.Image('USDA/NASS/CDL/2008');

    # again, homebrew argmax
    mask = irr.select("irr").gt(irr.select("unirr"));
    mask = mask.multiply(irr.select("irr").gt(irr.select("uncult")))

    mask = cdl.eq(cdl_class);
    irrmask = irr.select("irr").gt(irr.select("unirr"));
    counties = ee.FeatureCollection(county_shapefile)
    irrmask = irrmask.multiply(irr.select("irr").gt(irr.select("uncult")));

    irrcropmask = irrmask.multiply(mask);
    unirrcropmask = mask.multiply(irrmask.Not());

    # pixelArea is an ee builtin
    area_raster = ee.Image.pixelArea();

    irrigated_area_raster = area_raster.multiply(irrcropmask)
    unirrigated_area_raster = area_raster.multiply(unirrcropmask)
    def clip_to_counties_irr(feature): 
        name = feature.get("NAME")
        return irrigated_area_raster.clip(feature).set("NAME", name)

    def clip_to_counties_unirr(feature): 
        name = feature.get("NAME")
        return unirrigated_area_raster.clip(feature).set("NAME", name)

    irr_ic = ee.ImageCollection(counties.map(clip_to_counties_irr))
    unirr_ic = ee.ImageCollection(counties.map(clip_to_counties_unirr))

    def wrap(type_, include_geom):

        def reduce_area(image):

            areas = image.reduceRegion(
                  reducer=ee.Reducer.sum(),
                  geometry=image.geometry(),
                  crs="EPSG:5070",
                  scale=30,
                  maxPixels=1e9
                  )
            if include_geom:
                feature = ee.Feature(image.geometry(), {'AREA': areas.get('area'),
                    'NAME': image.get('NAME'), 'YEAR': year, 'TYPE':type_})
            else:
                feature = ee.Feature(None, {'AREA': areas.get('area'),
                    'NAME': image.get('NAME'), 'YEAR': year, 'TYPE':type_})

            return feature

        return reduce_area
    
    irr_reduce = wrap('IRR')
    unirr_reduce = wrap('UNIRR')

    irr_areas = irr_ic.map(irr_reduce)
    unirr_area = unirr_ic.map(unirr_reduce)

    return irr_areas.merge(unirr_area)


if __name__ == '__main__':
    #timeseries_met_data_mt(2008)
    '''
    irrigated_raster = 'users/tcolligan0/irrigation-rasters-sept27/irrMT{}'
    county_shapefile = 'users/tcolligan0/County'
    fc = ee.FeatureCollection([])
    dict_of_dicts = {}
    for year in range(2008, 2019):
        print(year)
        fc = irrigated_acres_by_crop(irrigated_raster.format(year), None, year)
        dict_of_dicts[str(year)] = fc

    import json
    with open('crop_proportions.json', 'w') as f:
        j = json.dumps(dict_of_dicts)
        f.write(j)


    '''
    for year in range(2000, 2020):
        raster = irrigated_raster.format(year)
        print('analyzing', os.path.basename(raster))
        feature_collection = export_timeseries_of_met_data(county_shapefile, 2012, reduction_type='mean')
        fc = fc.merge(feature_collection)

    task = ee.batch.Export.table.toDrive(
            collection=fc,
            description='precipallyears_10-06',
            )
    task.start()
    '''
    for year in range(2008, 2020):
        raster = irrigated_raster.format(year)
        print('analyzing', os.path.basename(raster))
        feature_collection = mask_irrigated_raster_to_cdl(raster, 36, county_shapefile, year)
        task = ee.batch.Export.table.toAsset(
                collection=feature_collection,
                description='shapefileAlf',
                assetId='users/tcolligan0/exampleExport',
                )
        task.start()
        exit()
        fc = fc.merge(feature_collection)

    '''
