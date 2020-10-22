import rasterio
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

fs = glob('/home/thomas/mt/montana-irr-rasters/rasters/*tif')

out = []
first = True
for f in fs:
    print(f)
    with rasterio.open(f, 'r') as src:
        meta = src.meta
        irr = src.read()
    if first:
        cnt = np.zeros((irr.shape[1], irr.shape[2]))
        cnt[np.argmax(irr, axis=0) == 0] += 1
        first = False
    else:
        cnt[np.argmax(irr, axis=0) == 0] += 1

meta.update({'count':1})
with rasterio.open('./irr_count.tif', 'w', **meta) as dst:
    dst.write(cnt.astype(np.uint8), 1)
