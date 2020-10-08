import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from argparse import ArgumentParser

import utils.utils as utils


def numpyfy(dct):

    out = {}
    for k, v in dct.items():
        out[k] = v.numpy()
    return out


if __name__ == '__main__':

    ap = ArgumentParser()

    ap.add_argument('--data-directory', required=True)
    ap.add_argument('--model-directory', required=True)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--n-classes', type=int, default=3)
    ap.add_argument('--use-cuda', action='store_true')
    ap.add_argument('--show-logs', action='store_true')
    ap.add_argument('--year', default=None, type=str)
    ap.add_argument('--add-ndvi', action='store_true')

    args = ap.parse_args()

    if not args.use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if not args.show_logs:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import tensorflow as tf

    model_path = args.model_directory
    model = args.data_directory

    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures['serving_default']

    dataset = utils.make_validation_dataset(args.data_directory,
                                            add_ndvi=args.add_ndvi, 
                                            batch_size=args.batch_size,
                                            year=args.year, 
                                            n_classes=args.n_classes,
                                            buffer_size=1,
                                            temporal_unet=False)

    # 2003, 2008, 2009, 2010, 2011, 2012, 2013, 2015
    if not isinstance(dataset, list):
        dataset = [dataset]

    c, p, r, i = utils.confusion_matrix_from_generator(dataset, 
                                                       batch_size=args.batch_size,
                                                       model=model,
                                                       n_classes=args.n_classes)

    print('model path:', model_path)
    print(c.numpy())
    print('------------')
    print(numpyfy(p))
    print('------------')
    print(numpyfy(r))
    print('------------')
    print(i)
