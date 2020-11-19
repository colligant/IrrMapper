import os
from argparse import ArgumentParser
from glob import glob


if __name__ == '__main__':



    dirs = ['montana-image-data']

    from evaluate_image import model_predictions
    for m in models:

        for d in dirs:

            outd = os.path.basename(m)
            outd = os.path.join('/home/thomas/share/results/', outd);

            if not os.path.isdir(outd):
                os.makedirs(outd, exist_ok=True)

            model_predictions(
                              model_path=m,
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
