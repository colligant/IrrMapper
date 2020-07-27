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
    ap.add_argument('--data-directory')
    ap.add_argument('--model-directory')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--job-dir', default=64)
    ap.add_argument('--yearly-stats', action='store_true')
    ap.add_argument('--n-classes', type=int)

    args = ap.parse_args()
    model_path = args.model_directory
    model = args.data_directory

    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures['serving_default']
    if args.yearly_stats:
        dataset_dict = utils.make_yearly_test_dataset(args.data_directory)
        for year, dataset in dataset_dict.items():
            print(year, type(dataset))
            c, p, r, i = utils.confusion_matrix_from_generator([dataset],
                    batch_size=args.batch_size, model=model, n_classes=args.n_classes)
            print(c)
            print('------------')
            print(p)
            print('------------')
            print(r)
            print('------------')
            print(i)
            print('------------')

    else:
        dataset = utils.make_test_dataset(args.data_directory, add_ndvi=False)
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
