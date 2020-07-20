import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import gc
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
import numpy as np
from pdb import set_trace
from scipy.ndimage import gaussian_filter
from glob import glob
from sys import stdout, exit
from rasterio import open as rasopen
import rasterio
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from models import unet
from random import shuffle

class AdamOptimizer:
    '''
    Since all of the feature visualization examples for tensorflow
    rely on updating the parameters of the image manually
    i.e.
    params -= steps_size*grads 
    I've implemented Adam for image optimization.
    The one thing to notice is that it does gradient ascent, not descent.
    '''

    def __init__(self, img_to_optimize, step_size, beta_1=0.9, beta_2=0.999, eps=1e-8):

        self.m = tf.zeros_like(img_to_optimize)
        self.v = tf.zeros_like(img_to_optimize)
        self.step_size = tf.constant(step_size)
        self.beta_1 = tf.constant(beta_1)
        self.beta_2 = tf.constant(beta_2)
        self.eps = tf.constant(eps)
        self.t = tf.constant(0.0)


    @tf.function(
        input_signature=(
          tf.TensorSpec(shape=[None, None, 30], dtype=tf.float32),
          tf.TensorSpec(shape=[None, None, 30], dtype=tf.float32),)
    )
    def step(self, img, grads):
        self.t += 1
        self.m = self.beta_1 * self.m + (1-self.beta_1)*grads
        self.v = self.beta_2 * self.v + (1-self.beta_2)*grads**2
        self.m = self.m / (1-self.beta_1**self.t)
        self.v = self.v / (1-self.beta_2**self.t)
        img += self.step_size * self.m / (tf.math.sqrt(self.v) + self.eps)
        return img


def calc_loss(channel_idx):

    def lfunc(img, model):
        img_batch = tf.expand_dims(img, 0)
        layer_act = model(img_batch)[0, :, :, channel_idx]
        layer_act = [layer_act]
        losses = []
        for act in layer_act:
            loss = tf.math.reduce_mean(act)
            losses.append(loss)
        return tf.reduce_sum(losses)
    
    return lfunc


class FeatVis(tf.Module):

    def __init__(self, model, loss_func, optim=None):
        self.model = model
        self.optim = optim
        self.loss_func = loss_func

    @tf.function(
        input_signature=(
          tf.TensorSpec(shape=[None,None,30], dtype=tf.float32),
          tf.TensorSpec(shape=[], dtype=tf.int32),
          tf.TensorSpec(shape=[], dtype=tf.float32),)
    )
    def __call__(self, img, steps, step_size):

        loss = tf.constant(0.0)
        for n in tf.range(steps):
          with tf.GradientTape() as tape:
            # This needs gradients relative to `img`
            # `GradientTape` only watches `tf.Variable`s by default
            tape.watch(img)
            loss = self.loss_func(img, self.model)

          # Calculate the gradient of the loss with respect to the pixels of the input image.
          gradients = tape.gradient(loss, img)

          if self.optim is None:
              img = img + gradients*step_size
          else:
              img = self.optim.step(img, gradients)
          # img = tf.clip_by_value(img, -1, 1)

        return loss, img

if __name__ == '__main__':
    model_path = './my_model.h5'

    model = tf.keras.models.load_model(model_path)

    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)
    # exit()
    # #vis_model = tf.keras.Model(inputs=model.input, outputs=model.layers[43].output)
    vis_model = tf.keras.Model(inputs=model.input, outputs=model.layers[76].output)
    for layer in vis_model.layers:
        layer.trainable = False

    for channel_idx in [2]:

        init_img = np.zeros((30, 32, 32))
        img = tf.convert_to_tensor(tf.cast(tf.transpose(init_img, [1, 2, 0]), tf.float32))

        step_size = tf.convert_to_tensor(1.0)
        optim = AdamOptimizer(img, step_size)
        loss_func = calc_loss(channel_idx)
        vis = FeatVis(vis_model, loss_func, optim)

        for i in range(1000):
            loss, img = vis(img, tf.constant(100), tf.constant(step_size))
            print(tf.reduce_mean(img).numpy())
            if i % 100 == 0 and i <= 500:
                img = gaussian_filter(img, sigma=2)

        for i in range(img.shape[2]):
            fig, ax = plt.subplots()
            ax.imshow(img[:, :, i])
            plt.show()
