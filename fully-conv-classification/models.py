import os
import tensorflow as tf
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu


def gradient_wrt_inputs(model, data):
    layer_output = model.output
    loss = -tf.reduce_mean(layer_output)
    grads = K.gradients(loss, model.input[0])[0]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    weights = np.ones((1, 388, 388, 5))
    results = sess.run(grads, feed_dict={model.input[0]:data, model.input[1]:weights})
    return results

_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)


def conv_lstm_3d(input_shape, n_classes=3):
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(8, 100, 100, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(MaxPooling3D())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(UpSampling3D())

    seq.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(Conv3D(filters=3, kernel_size=(1, 1, 1),
                   activation=None,
                   padding='same', data_format='channels_last'))

    return seq


def small_lstm_unet_fc_layer(input_shape):

    inp = Input(input_shape)
    exponent = 2
    base = 4

    c1 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(inp)
    c2 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c1)
    b1 = TimeDistributed(BatchNormalization())(c2)
    d1 = TimeDistributed(MaxPooling2D())(b1)

    exponent += 1

    c3 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d1)
    c4 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c3)
    b2 = TimeDistributed(BatchNormalization())(c4)
    d2 = TimeDistributed(MaxPooling2D())(b2)

    exponent += 1

    c5 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d2)
    c6 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c5)
    b3 = TimeDistributed(BatchNormalization())(c6)
    u2 = TimeDistributed(UpSampling2D())(b3)
    concat2 = Concatenate()([u2, c4])

    flatten = TimeDistributed(Flatten())(concat2)

    fc = TimeDistributed(Dense(units=128, activation='relu'))(flatten)
    out = TimeDistributed(Dense(units=3, activation='softmax'))(fc)

    model = Model(inputs=inp, outputs=out)
    return model

def small_unet_residual_connections(input_shape):

    inp = Input(input_shape)
    exponent = 2
    base = 5

    c1 = ResBlock(inp, filters=base**exponent, snd_conv=True)
    c2 = ResBlock(c1, filters=base**exponent)
    d1 = MaxPooling2D()(c2)

    exponent += 1

    c3 = ResBlock(d1, filters=base**exponent, snd_conv=True)
    c4 = ResBlock(c3, filters=base**exponent)
    d2 = MaxPooling2D()(c4)

    exponent += 1

    c5 = ResBlock(d2, filters=base**exponent)
    c6 = ResBlock(c5, filters=base**exponent)
    u1 = UpSampling2D()(c6)
    concat1 = Concatenate()([u1, c4])

    exponent -= 1

    c7 = ResBlock(concat1, filters=base**exponent)
    c8 = ResBlock(c7, filters=base**exponent)
    u2 = UpSampling2D()(c8)
    concat2 = Concatenate()([u2, c2])

    exponent -= 1
    c9 = ResBlock(concat2, filters=base**exponent)
    c10 = ResBlock(c9, filters=base**exponent)

    conv_out = Conv2D(filters=3, kernel_size=1, padding='same', activation='softmax')(c10)
    model = Model(inputs=inp, outputs=conv_out)

    return model


def small_unet_smarter(input_shape, base=4):

    inp = Input(input_shape)
    exponent = 2
    base = base

    c1 = ConvBNRelu(inp, filters=base**exponent)
    c2 = ConvBNRelu(c1, filters=base**exponent)
    d1 = MaxPooling2D()(c2)

    exponent += 1

    c3 = ConvBNRelu(d1, filters=base**exponent)
    c4 = ConvBNRelu(c3, filters=base**exponent)
    d2 = MaxPooling2D()(c4)

    exponent += 1

    c5 = ConvBNRelu(d2, filters=base**exponent)
    c6 = ConvBNRelu(c5, filters=base**exponent)
    u1 = UpSampling2D()(c6)
    concat1 = Concatenate()([u1, c4])

    exponent -= 1

    c7 = ConvBNRelu(concat1, filters=base**exponent)
    c8 = ConvBNRelu(c7, filters=base**exponent)
    u2 = UpSampling2D()(c8)
    concat2 = Concatenate()([u2, c2])

    exponent -= 1
    c9 = ConvBNRelu(concat2, filters=base**exponent)
    c10 = ConvBNRelu(c9, filters=base**exponent)

    conv_out = Conv2D(filters=3, kernel_size=1, padding='same', activation='softmax')(c10)
    model = Model(inputs=inp, outputs=conv_out)

    return model


def small_unet(input_shape):

    inp = Input(input_shape)
    exponent = 2
    base = 5

    c1 = Conv2D(filters=base**exponent, kernel_size=(3, 3),
            activation='relu', padding='same')(inp)
    c2 = Conv2D(filters=base**exponent, kernel_size=(3, 3),
            activation='relu', padding='same')(c1)
    b1 = BatchNormalization()(c2)
    d1 = MaxPooling2D()(b1)

    exponent += 1

    c3 = Conv2D(filters=base**exponent, kernel_size=(3, 3), activation='relu', padding='same')(d1)
    c4 = Conv2D(filters=base**exponent, kernel_size=(3, 3), activation='relu', padding='same')(c3)
    b2 = BatchNormalization()(c4)
    d2 = MaxPooling2D()(b2)

    exponent += 1

    c5 = Conv2D(filters=base**exponent, kernel_size=(3, 3), activation='relu', padding='same')(d2)
    c6 = Conv2D(filters=base**exponent, kernel_size=(3, 3), activation='relu', padding='same')(c5)
    b3 = BatchNormalization()(c6)
    u2 = UpSampling2D()(b3)
    concat2 = Concatenate()([u2, c4])

    exponent -= 1

    c7 = Conv2D(filters=base**exponent, kernel_size=(3, 3), activation='relu',
            padding='same')(concat2)
    c8 = Conv2D(filters=base**exponent, kernel_size=(3, 3), activation='relu', padding='same')(c7)
    b4 = BatchNormalization()(c8)
    u3 = UpSampling2D()(b4)
    concat3 = Concatenate()([u3, c2])

    exponent -= 1

    c13 = Conv2D(filters=base**exponent, kernel_size=(3, 3), activation='relu',
            padding='same')(concat3)
    c14 = Conv2D(filters=base**exponent, kernel_size=(3, 3), activation='relu', padding='same')(c13)
    conv_out = Conv2D(filters=3, kernel_size=1, padding='same', activation='softmax')(c14)
    model = Model(inputs=inp, outputs=conv_out)

    return model

def small_lstm_unet(input_shape, base_filters=4):

    inp = Input(input_shape)
    exponent = 2
    base = base_filters

    c1 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(inp)
    c2 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c1)
    b1 = TimeDistributed(BatchNormalization())(c2)
    d1 = TimeDistributed(MaxPooling2D())(b1)

    exponent += 1

    c3 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d1)
    c4 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c3)
    b2 = TimeDistributed(BatchNormalization())(c4)
    d2 = TimeDistributed(MaxPooling2D())(b2)

    exponent += 1

    c5 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d2)
    c6 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c5)
    b3 = TimeDistributed(BatchNormalization())(c6)
    u2 = TimeDistributed(UpSampling2D())(b3)
    concat2 = Concatenate()([u2, c4])

    exponent -= 1

    c7 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
          return_sequences=True)(concat2)
    c8 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(c7)
    b4 = TimeDistributed(BatchNormalization())(c8)
    u3 = TimeDistributed(UpSampling2D())(b4)
    concat3 = Concatenate()([u3, c2])

    exponent -= 1

    c13 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(concat3)
    c14 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(c13)
    conv_out = TimeDistributed(Conv2D(filters=3, kernel_size=1,
        padding='same', activation='softmax'))(c14)
    #c14 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
    #        return_sequences=False)(c13)
    #conv_out = Conv2D(filters=3, kernel_size=1,
    #    padding='same', activation='softmax')(c14)

    model = Model(inputs=inp, outputs=conv_out)
    return model


def non_recurrent_decoder(input_shape):

    inp = Input(input_shape)
    exponent = 2
    base = 4

    c1 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(inp)
    c2 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c1)
    b1 = TimeDistributed(BatchNormalization())(c2)
    d1 = TimeDistributed(MaxPooling2D())(b1)

    exponent += 1

    c3 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d1)
    c4 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c3)
    b2 = TimeDistributed(BatchNormalization())(c4)
    d2 = TimeDistributed(MaxPooling2D())(b2)

    exponent += 1

    c5 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d2)
    c6 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c5)
    b3 = TimeDistributed(BatchNormalization())(c6)
    u2 = TimeDistributed(UpSampling2D())(b3)
    concat2 = Concatenate()([u2, c4])

    exponent -= 1

    c7 = TimeDistributed(Conv2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
          activation='relu'))(concat2)
    c8 = TimeDistributed(Conv2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            activation='relu'))(c7)
    b4 = TimeDistributed(BatchNormalization())(c8)
    u3 = TimeDistributed(UpSampling2D())(b4)
    concat3 = Concatenate()([u3, c2])

    exponent -= 1

    c13 = TimeDistributed(Conv2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            activation='relu'))(concat3)
    c14 = TimeDistributed(Conv2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            activation='relu'))(c13)


    conv_out = TimeDistributed(Conv2D(filters=3, kernel_size=1,
        padding='same', activation='softmax'))(c14)

    model = Model(inputs=inp, outputs=conv_out)

    return model 


def lstm_unet(input_shape):

    inp = Input(input_shape)
    exponent = 2
    base = 3

    c1 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(inp)
    c2 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c1)
    d1 = TimeDistributed(MaxPooling2D())(c2)

    exponent += 1

    c3 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d1)
    c4 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c3)
    d2 = TimeDistributed(MaxPooling2D())(c4)


    c5 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d2)
    c6 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c5)
    d3 = TimeDistributed(MaxPooling2D())(c6)


    c7 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(d3)
    c8 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same',
            return_sequences=True)(c7)

    u1 = TimeDistributed(UpSampling2D())(c8)
    concat1 = Concatenate()([u1, c6])


    c9 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(concat1)
    c10 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(c9)

    u2 = TimeDistributed(UpSampling2D())(c10)
    concat2 = Concatenate()([u2, c4])

    c11 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(concat2)
    c12 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(c11)

    exponent -= 1 
    u3 = TimeDistributed(UpSampling2D())(c12)
    concat3 = Concatenate()([u3, c2])

    c13 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(concat3)
    c14 = ConvLSTM2D(filters=base**exponent, kernel_size=(3, 3), padding='same', 
            return_sequences=True)(c13)


    conv_out = TimeDistributed(Conv2D(filters=3, kernel_size=1,
        padding='same', activation='softmax'))(c14)

    model = Model(inputs=inp, outputs=conv_out)

    return model 

def conv_lstm_2d(input_shape, n_classes=3):
    seq = Sequential()
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(8, 100, 100, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(TimeDistributed(MaxPooling2D()))

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(TimeDistributed(MaxPooling2D()))

    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())
    seq.add(ConvLSTM2D(filters=64, kernel_size=(3, 3),
                       padding='same', return_sequences=True))

    seq.add(TimeDistributed(UpSampling2D()))

    seq.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
        padding='same')))
    seq.add(BatchNormalization())

    seq.add(TimeDistributed(Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
        padding='same')))
    seq.add(BatchNormalization())

    seq.add(TimeDistributed(Conv2D(filters=3, kernel_size=(1, 1),
                   activation=None,
                   padding='same', data_format='channels_last')))
    return seq


def ConvBlock(x, filters=64, padding='same'):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding=padding,
            kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding=padding,
        kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    return Activation(relu)(x)


def ResBlock(x, filters=64, snd_conv=False):
    x1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.01))(x)
    x1 = BatchNormalization()(x1)
    x1 = Activation(relu)(x1)
    if snd_conv:
        x1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
                kernel_regularizer=l2(0.01))(x1)
        x1 = Concatenate()([x1, x])
        x1 = BatchNormalization()(x1)
        x1 = Activation(relu)(x1)
    return x1


def ConvBNRelu(x, filters=64):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',
            kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    return Activation(relu)(x)


_epsilon = tf.convert_to_tensor(K.epsilon(), tf.float32)

def two_headed_unet(input_shape, initial_exp=6, n_classes=5):
     
    features = Input(shape=input_shape)
    _power = initial_exp
    exp = 2

    c1 = ConvBlock(features, exp**_power)
    mp1 = MaxPooling2D(pool_size=2, strides=2)(c1)

    _power += 1

    c2 = ConvBlock(mp1, exp**_power)
    mp2 = MaxPooling2D(pool_size=2, strides=2)(c2)

    _power += 1

    c3 = ConvBlock(mp2, exp**_power)
    mp3 = MaxPooling2D(pool_size=2, strides=2)(c3)

    _power += 1 

    c4 = ConvBlock(mp3, exp**_power)
    mp4 = MaxPooling2D(pool_size=2, strides=2)(c4)

    _power += 1

    # 1024 filters
    c5 = ConvBlock(mp4, exp**_power)
    _power -= 1

    u1 = UpSampling2D(size=(2, 2))(c5)
    c6 = ConvBNRelu(u1, filters=exp**_power)
    u1_c4 = Concatenate()([c6, c4])
    c7 = ConvBlock(u1_c4, filters=exp**_power)

    _power -= 1
    
    u2 = UpSampling2D(size=(2, 2))(c7)
    c8 = ConvBNRelu(u2, filters=exp**_power)
    u2_c3 = Concatenate()([c8, c3])
    c9 = ConvBlock(u2_c3, filters=exp**_power)

    _power -= 1
    
    u3 = UpSampling2D(size=(2, 2))(c9)
    c10 = ConvBNRelu(u3, filters=exp**_power)
    u3_c2 = Concatenate()([c10, c2])
    c11 = ConvBlock(u3_c2, filters=exp**_power)

    _power -= 1
    u4 = UpSampling2D(size=(2, 2))(c11)
    c12 = ConvBNRelu(u4, filters=exp**_power)
    u4_c1 = Concatenate()([c12, c1])
    c13 = ConvBlock(u4_c1, filters=exp**_power)
    cdl_logits = Conv2D(filters=1, kernel_size=1, strides=1,
                    activation=None, name='cdl')(c13)

    c14 = ConvBlock(cdl_logits, exp**_power)
    mp5 = MaxPooling2D(pool_size=2, strides=2)(c14)

    c15 = ConvBlock(c14, exp**_power)
    u5 = UpSampling2D(size=(2, 2))(c15)
    concat_final = Concatenate()([c15, c13])

    irr_logits = Conv2D(filters=n_classes, kernel_size=1, strides=1,
                    activation=None, name='irr')(concat_final)
    
    return Model(inputs=[features], outputs=[irr_logits, cdl_logits])


def fcnn(input_shape):
    features = Input(shape=input_shape)

    c1 = ConvBlock(features, 64)
    c1 = ConvBlock(c1, 64)

    irr_logits = Conv2D(filters=3, kernel_size=1, strides=1,
                    activation=None, name='irr')(c1)

    return Model(inputs=[features], outputs=[irr_logits])



def unet(input_shape, initial_exp=6, n_classes=5):
     
    features = Input(shape=input_shape)
    base = 2

    c1 = ConvBlock(features, base**initial_exp)
    mp1 = MaxPooling2D(pool_size=2, strides=2)(c1)

    initial_exp += 1

    c2 = ConvBlock(mp1, base**initial_exp)
    mp2 = MaxPooling2D(pool_size=2, strides=2)(c2)

    initial_exp += 1

    c3 = ConvBlock(mp2, base**initial_exp)
    mp3 = MaxPooling2D(pool_size=2, strides=2)(c3)

    initial_exp += 1 

    c4 = ConvBlock(mp3, base**initial_exp)
    mp4 = MaxPooling2D(pool_size=2, strides=2)(c4)

    initial_exp += 1

    # 1024 filters
    c5 = ConvBlock(mp4, base**initial_exp)
    initial_exp -= 1

    u1 = UpSampling2D(size=(2, 2))(c5)
    c6 = ConvBNRelu(u1, filters=base**initial_exp)
    u1_c4 = Concatenate()([c6, c4])
    c7 = ConvBlock(u1_c4, filters=base**initial_exp)

    initial_exp -= 1
    
    u2 = UpSampling2D(size=(2, 2))(c7)
    c8 = ConvBNRelu(u2, filters=base**initial_exp)
    u2_c3 = Concatenate()([c8, c3])
    c9 = ConvBlock(u2_c3, filters=base**initial_exp)

    initial_exp -= 1
    
    u3 = UpSampling2D(size=(2, 2))(c9)
    c10 = ConvBNRelu(u3, filters=base**initial_exp)
    u3_c2 = Concatenate()([c10, c2])
    c11 = ConvBlock(u3_c2, filters=base**initial_exp)

    initial_exp -= 1
    u4 = UpSampling2D(size=(2, 2))(c11)
    c12 = ConvBNRelu(u4, filters=base**initial_exp)
    u4_c1 = Concatenate()([c12, c1])
    c13 = ConvBlock(u4_c1, filters=base**initial_exp)

    logits = Conv2D(filters=n_classes, kernel_size=1, strides=1,
                    activation='softmax', name='logits')(c13)
    
    return Model(inputs=[features], outputs=[logits])

if __name__ == '__main__':
    mm = small_unet((None, None, 39))
    mm2 = small_lstm_unet((None, None, None, 39))
    print('hi')
    mm.summary()
    mm.summary()
    mm2.summary()
