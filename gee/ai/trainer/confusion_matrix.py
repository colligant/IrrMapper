import tensorflow as tf

from argparse import ArgumentParser
from .utils import confusion_matrix_from_generator, make_test_dataset


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument('--data-directory')
    ap.add_argument('--model-directory')
    ap.add_argument('--batch-size', default=64)
    ap.add_argument('--job-dir', default=64)

    args = ap.parse_args()
    model_path = args.model_directory
    model = args.data_directory


    dataset = make_test_dataset(args.data_directory)
    if not isinstance(dataset, list):
        dataset = [dataset]


    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures['serving_default']

    c, p, r, i = confusion_matrix_from_generator(dataset, batch_size=args.batch_size, model=model)

    print('model path:', model_path)
    print(c)
    print('------------')
    print(p)
    print('------------')
    print(r)
    print('------------')
    print(i)
