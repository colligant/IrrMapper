import os
import tensorflow as tf
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from pdb import set_trace
from glob import glob
from sys import stdout, exit
from tensorflow.keras.models import load_model

from evaluate_image import model_predictions
from argparse import ArgumentParser



if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--model-path', required=True)
    args = ap.parse_args()
    dirs = ['image-data-2016-and-on', 'leftover-image-data',
            'montana-image-data']
    for d in dirs:

        outd = os.path.basename(args.model_path)
        outd = os.path.join('/home/thomas/ssd/', outd);
        if not os.path.isdir(outd):
            os.makedirs(outd, exist_ok=True)

        model_predictions(
                model_path=args.model_path,
                data_directory=os.path.join('/home/thomas/share/', d),
                image_file=None,
                year=None,
                out_directory=outd,
                n_classes=3,
                use_cuda=True,
                tile_size=608,
                chunk_size=512,
                show_logs=True,
                ndvi=False,
                dropout=False)
