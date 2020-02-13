import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16

# use VGG as encoder since we're working with rgb
encoder = Sequential()
encoder.add(kl.Conv2D(16, 3, input_shape=(120, 60, 1)))
encoder.add(kl.Conv2D(16, 3))
encoder.add(kl.MaxPooling2D())
encoder.add(kl.Conv2D(16, 3))
encoder.add(kl.Conv2D(16, 3))
encoder.add(kl.Flatten()) # Not sure if this if the proper way to do this.

rnn = Sequential()
rnn = kl.GRU(64, return_sequences=False, input_shape=(120, 60))

decoder = Sequential()
decoder.add(kl.Reshape((60, 30, 16)))
decoder.add(kl.Conv2D(16, 3))
decoder.add(kl.Conv2D(16, 3))
decoder.add(kl.UpSampling2D())
decoder.add(kl.Conv2D(16, 3))
decoder.add(kl.Conv2D(16, 3))
decoder.add(kl.Conv2D(4, 1))

inp = kl.Input(shape=(5, 120, 60, 1))
model = kl.TimeDistributed(encoder)(inp)
model = rnn(model)
model = decoder(model)

model = Model(inputs=inp, outputs=model)

model.summary()

