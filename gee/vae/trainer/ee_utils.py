import ee

def temporalCollection(collection, start, count, interval, units):
 
  sequence = ee.List.sequence(0, ee.Number(count).subtract(1))
  originalStartDate = ee.Date(start)
  def filt(i):
    
    startDate = originalStartDate.advance(ee.Number(interval).multiply(i), units)

    endDate = originalStartDate.advance(
        ee.Number(interval).multiply(ee.Number(i).add(1)), units)
    return collection.filterDate(startDate, endDate).reduce(ee.Reducer.mean())
  return ee.ImageCollection(sequence.map(filt))


def extract_data_over_polygons(polygon_list, data_stack, out_folder, file_basename, features,
                               n_shards=10, every_n=1):
  geomSample = ee.ImageCollection([])
  len_geom_sample = 0
  for i, g in enumerate(range(polygon_list.size().getInfo())):
    if i % every_n != 0:
        continue
    sample = data_stack.sample(
      region = ee.Feature(polygon_list.get(g)).geometry(), 
      scale = 30,
      numPixels = 1, # Size of the shard.
      seed = i,
      tileScale = 8
      )
    geomSample = geomSample.merge(sample)
    len_geom_sample += 1
    if len_geom_sample == n_shards:
      desc = file_basename + '_g' + str(g)
      print('saving to {}'.format(out_folder))
      task = ee.batch.Export.table.toDrive(
        collection=geomSample,
        description=desc,
        fileFormat='TFRecord',
        folder=out_folder,
        fileNamePrefix = file_basename + str(time.time()),
        selectors=features
        )
      task.start()
      geomSample = ee.ImageCollection([])
      len_geom_sample = 0


def assign_class_code(shapefile_path):
  if 'irrigated' in shapefile_path and 'unirrigated' not in shapefile_path:
      return 0
  if 'unirrigated' in shapefile_path:
      return 1
  if 'wetlands' in shapefile_path or 'uncultivated' in shapefile_path:
      return 2
  else:
      raise NameError('shapefile path {} isn\'t named in assign_class_code'.format(shapefile_path))
