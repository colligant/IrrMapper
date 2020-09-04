import os
import tensorflow as tf

from argparse import ArgumentParser
from . import utils 

def numpyfy(dct):

    out = {}
    for k, v in dct.items():
        out[k] = v.numpy()
    return out


if __name__ == '__main__':

    ap = ArgumentParser()

    ap.add_argument('--job-dir')
    ap.add_argument('--data-directory')
    ap.add_argument('--model-directory')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--n-classes', type=int)
    ap.add_argument('--use-cuda', action='store_true')
    ap.add_argument('--year', default=None, type=str)
    ap.add_argument('--add-ndvi', action='store_true')

    args = ap.parse_args()

    if not args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model_path = args.model_directory
    model = args.data_directory

    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures['serving_default']

    dataset = utils.make_validation_dataset(args.data_directory, add_ndvi=args.add_ndvi, 
            batch_size=args.batch_size,
            year=args.year, 
            n_classes=args.n_classes)

    if not isinstance(dataset, list):
        dataset = [dataset]

    c, p, r, i = utils.confusion_matrix_from_generator(dataset, 
            batch_size=args.batch_size, model=model, n_classes=args.n_classes)

    print('model path:', model_path)
    print(c.numpy())
    print('------------')
    print(numpyfy(p))
    print('------------')
    print(numpyfy(r))
    print('------------')
    print(i)
