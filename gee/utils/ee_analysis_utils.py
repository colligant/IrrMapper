import ee
import os
ee.Initialize()


ACRES_PER_SQUARE_METER = 0.000247105

def confusion_matrices(irr_labels, unirr_labels, irr_image, unirr_image):
    mt = "users/tcolligan0/Montana"
    mt = ee.FeatureCollection(mt)
    mt = mt.toList(mt.size()).get(0)
    mt = ee.Feature(mt)

    true_positive = irr_image.eq(irr_labels) # pred. irrigated, labeled irrigated
    false_positive = irr_image.eq(unirr_labels) # pred. irrigated, labeled unirrigated

    true_negative = unirr_image.eq(unirr_labels) # pred unirrigated, labeled unirrigated
    false_negative = unirr_image.eq(irr_labels) # pred unirrigated, labeled irrigated

    TP = true_positive.reduceRegion(
     geometry=mt.geometry(),
     reducer=ee.Reducer.count(),
     maxPixels=1e9,
     crs='EPSG:5070',
     scale=30
     )
    FP = false_positive.reduceRegion(
     geometry=mt.geometry(),
     reducer=ee.Reducer.count(),
     maxPixels=1e9,
     crs='EPSG:5070',
     scale=30
     )
    FN = false_negative.reduceRegion(
     geometry=mt.geometry(),
     reducer=ee.Reducer.count(),
     maxPixels=1e9,
     crs='EPSG:5070',
     scale=30
     )
    TN = true_negative.reduceRegion(
     geometry=mt.geometry(),
     reducer=ee.Reducer.count(),
     maxPixels=1e9,
     crs='EPSG:5070',
     scale=30
     )

    print("TP", TP.getInfo())
    print("FP", FP.getInfo())
    print("FN", FN.getInfo())
    print("TN", TN.getInfo())


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


def create_lanid_labels(year):
    lanid = ee.Image('users/xyhisu/irrigationMapping/results/LANID12');
    irr_mask = lanid.eq(1); # lanid is already masked

    unmasked = lanid.unmask(0);

    unirr_image = ee.Image(1).byte().updateMask(unmasked.Not());
    irr_image = ee.Image(1).byte().updateMask(irr_mask);
    return irr_image, unirr_image

def create_unet_labels(year):
    unet = ee.Image("users/tcolligan0/irrigationMT/irrMT{}".format(year))
    unet = unet.select(["b1", "b2", "b3"], ["irr", "uncult", "unirr"])
    irrImage = unet.select("irr")
    irrMask = irrImage.gt(unet.select('uncult'))
    irrMask = irrMask.And(unet.select('irr').gt(unet.select('unirr')))
    unirrImage = ee.Image(1).byte().updateMask(irrMask.neq(1))
    irrImage = ee.Image(1).byte().updateMask(irrMask.eq(1))
    return irrImage, unirrImage

def create_rf_labels(year):
    year = '2012';
    begin = year + '-01-01'
    end = year + '-12-31'
    rf = ee.ImageCollection('users/dgketchum/IrrMapper/version_2');
    rf = rf.filter(ee.Filter.date(begin, end)).select('classification').mosaic();
    irrMask = rf.lt(1);
    unirrImage = ee.Image(1).byte().updateMask(irrMask.Not());
    irrImage = ee.Image(1).byte().updateMask(irrMask);
    return irrImage, unirrImage

def create_mirad_labels(year):
    mirad = ee.Image('users/tcolligan0/MIRAD/mirad{}MT'.format(year));
    irrMask = mirad.eq(1);
    unirrImage = ee.Image(1).byte().updateMask(irrMask.Not());
    irrImage = ee.Image(1).byte().updateMask(irrMask);
    return irrImage, unirrImage

def create_irrigated_labels(all_data, year):
    if all_data:
        non_irrigated = ee.FeatureCollection('users/tcolligan0/merged_shapefile_unirr_wetlands_unc');
        fallow = ee.FeatureCollection('users/tcolligan0/fallow_11FEB');
        irrigated = ee.FeatureCollection('users/tcolligan0/irrigated_MT_13MAR2020');
        fallow = fallow.filter(ee.Filter.eq("YEAR", year));
        non_irrigated = non_irrigated.merge(fallow);
        irrigated = irrigated.filter(ee.Filter.eq("YEAR", year));
    else:
        root = 'users/tcolligan0/test-data-aug24/'
        non_irrigated = ee.FeatureCollection(root + 'uncultivated_test');
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'unirrigated_test'));
        non_irrigated = non_irrigated.merge(ee.FeatureCollection(root + 'wetlands_buffered_test'));
        
        fallow = ee.FeatureCollection(root + 'fallow_test');
        irrigated = ee.FeatureCollection(root + 'irrigated_test');
        fallow = fallow.filter(ee.Filter.eq("YEAR", year));
        non_irrigated = non_irrigated.merge(fallow);
        irrigated = irrigated.filter(ee.Filter.eq("YEAR", year))

    irr_labels = ee.Image(1).byte().paint(irrigated, 0);
    irr_labels = irr_labels.updateMask(irr_labels.Not());
    unirr_labels = ee.Image(1).byte().paint(non_irrigated, 0)
    unirr_labels = unirr_labels.updateMask(unirr_labels.Not());

    return irr_labels, unirr_labels

if __name__ == '__main__':

    irr_base = 'users/tcolligan0/irrigation-rasters-sept27/irrMT{}'
    county = 'users/tcolligan0/County'
    fc = None
    for year in range(2000, 2020):
        irr = irr_base.format(year)
        fw = irrigated_predictions_by_county(irr, county, year)
        if fc is None:
            fc = fw
        else:
            fc = fc.merge(fw)
    task = ee.batch.Export.table.toDrive(
            collection=fc,
            description="all_years"
        )
    task.start()




    
