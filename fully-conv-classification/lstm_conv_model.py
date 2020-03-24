import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications.vgg16 import VGG16

def model():
    encoder = Sequential()
    encoder.add(kl.Conv2D(32, 3, input_shape=(100, 100, 3), padding='same',
        activation='relu'))
    encoder.add(kl.Conv2D(32, 3, padding='same', activation='relu'))
    encoder.add(kl.MaxPooling2D())
    encoder.add(kl.Conv2D(64, 3, padding='same', activation='relu'))
    encoder.add(kl.Conv2D(64, 3, padding='same', activation='relu'))

    ann = Sequential()
    ann = kl.ConvLSTM2D(filters=64, kernel_size=(3,3),
            return_sequences=True, padding='same', input_shape=(None, None, None, 3))
    ann = kl.ConvLSTM2D(filters=64, kernel_size=(3,3),
            return_sequences=False, padding='same')
    ann = kl.ConvLSTM2D(filters=64, kernel_size=(3,3),
            return_sequences=False, padding='same', input_shape=(None, None, None, 3))

    decoder = Sequential()
    decoder.add(kl.Conv2D(64, 3, padding='same', activation='relu'))
    decoder.add(kl.Conv2D(64, 3, padding='same', activation='relu'))
    decoder.add(kl.UpSampling2D())
    decoder.add(kl.Conv2D(32, 3, padding='same', activation='relu'))
    decoder.add(kl.Conv2D(32, 3, padding='same', activation='relu'))
    decoder.add(kl.Conv2D(3, 1, padding='same', activation='softmax'))

    inp = kl.Input(shape=(None, None, None, 3))
    model = kl.TimeDistributed(encoder)(inp)
    model = ann(model)
    model = decoder(model)
    model = Model(inputs=inp, outputs=model)
    return model
