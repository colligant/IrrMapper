import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2
import rasterio
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.python.framework import ops
from vae_copy.trainer.utils import make_balanced_training_dataset

import feature_spec

def build_model():
    model = tf.keras.models.load_model('/home/thomas/IrrMapper/gee/transformed_model.h5')
    return model

def build_guided_model():
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)

    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_idx, channel_idx):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.layers[layer_idx].output[0, :, :, channel_idx]
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val

if __name__ == '__main__':
    model = build_model()
    guided = build_guided_model()
    
    train = make_balanced_training_dataset('/home/thomas/ssd/vae-data/ks32/')
    for x_batch in train:
        preds = model.predict(arr, steps=1)
        for i in range(x_batch.shape[0]):
            channel_idx = 0
            grads_val = guided_backprop(model, x_batch[i], -1, channel_idx).squeeze()
            means = np.mean(grads_val, axis=(1, 2))
            plt.plot(means)
            plt.show()
            # fig, ax = plt.subplots(ncols=3)
            # grads_val /= np.max(grads_val)
            # for i in range(grads_val.shape[-1]):
            #     print(np.mean(grads_val[0, :, :, i]))
            #     ax[0].imshow(grads_val[0, :, :, i])
            #     ax[1].imshow(arr[0, :, :, i].eval(session=tf.compat.v1.Session()))
            #     ax[2].imshow(preds[0, :, :])
            #     plt.title(i)
            #     plt.pause(0.5)
            #     plt.cla()
            # plt.close()
