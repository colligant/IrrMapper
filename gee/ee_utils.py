import ee
import os

def temporalCollection(collection, start, count, interval, units):
 
  sequence = ee.List.sequence(0, ee.Number(count).subtract(1))
  originalStartDate = ee.Date(start)
  def filt(i):
    
    startDate = originalStartDate.advance(ee.Number(interval).multiply(i), units)

    endDate = originalStartDate.advance(
        ee.Number(interval).multiply(ee.Number(i).add(1)), units)
    return collection.filterDate(startDate, endDate).reduce(ee.Reducer.mean())
  return ee.ImageCollection(sequence.map(filt))

def assign_class_code(shapefile_path):
  shapefile_path = os.path.basename(shapefile_path)
  if 'irrigated' in shapefile_path and 'unirrigated' not in shapefile_path:
      return 0
  if 'unirrigated' in shapefile_path:
      return 1
  if 'fallow' in shapefile_path:
      return 2
  if 'wetlands' in shapefile_path:
      return 3
  if 'uncultivated' in shapefile_path:
      return 4
  if 'points':
      # annoying workaround for earthengine
      return 10
  else:
      raise NameError('shapefile path {} isn\'t named in assign_class_code'.format(shapefile_path))

if __name__ == '__main__':
    pass



