import os
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

TOKEN_SELF_ATTN_VALUE = 5e-4

def convblock(x, filters, weight_decay_const, apply_batchnorm, padding='same'):
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding=padding,
            kernel_regularizer=l2(weight_decay_const))(x)
    if apply_batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding=padding,
            kernel_regularizer=l2(weight_decay_const))(x)
    if apply_batchnorm:
        x = BatchNormalization()(x)
    return Activation('relu')(x)

def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x,  ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)


class TemporalAttention(tf.keras.layers.Layer):
    '''
    Temporal attention for embedded representations
    of satellite image tiles.
    '''

    def __init__(self, timesteps, input_shape, weight_decay_const):
        super(TemporalAttention, self).__init__()
        self.timesteps = timesteps
        self.concat = Concatenate()
        self.b, self.inp_h, self.inp_w, self.inp_d = input_shape

        self.dk = tf.convert_to_tensor(self.inp_h * self.inp_w * self.inp_d, tf.int32)
        self.sdk = tf.math.sqrt(tf.cast(self.dk, tf.float32))

        self.in_reshape = Reshape((self.inp_d*self.inp_h*self.inp_w, timesteps))
        self.out_reshape = Reshape((self.inp_h, self.inp_w, self.inp_d*timesteps))

        self.softmax_attention = Softmax()


    def __call__(self, queries, values):
        '''
        inputs: tuple of embedded 2D representations
        n x n x filters
        Uses shared QK attention, from Reformer (2020).
        QK attention requires that k_j = q_j / ||q_j||,
        and that tokens don't attend to themselves.
        (A dot product of a vector with a normalized version of  itself
         will be greater than the dot of it with all others, meaning
         ...attend to self?).
        '''

        queries = self.concat(queries)
        values = self.concat(values)
        q = self.in_reshape(queries)
        v = self.in_reshape(values)

        qk = make_unit_length(q)
        qkt = self.softmax_attention(tf.matmul(q, tf.transpose(qk, [0, 2, 1])) / self.sdk)
        # mask out diagonal for non-self attention. We don't
        # have to worry about the part "except where no
        # other context is available", because we're not decoding a sequence,
        # just using this temporal attention layer to incorporate time information
        # into a n-channel semantic segmentation mask.
        # masking out the diagonal is a more complex task in tensorflow.
        # I want: qkt[:, range(dk), range(dk)] = TOKEN_SELF_ATTN_VALUE,
        # but for some reason tf tensors don't support arbitrary indexing with
        qkt = tf.where(qkt == tf.transpose(qkt, [0, 2, 1]),
                tf.zeros_like(qkt)+TOKEN_SELF_ATTN_VALUE, qkt) 
        # ok, a sloppy solution. The diagonal of a matrix will be the same
        # even if it's tranposed. So that's what I do: ask where the transpose
        # is equal to the original matrix, then replace those values with
        # the masked value, and the rest of the values I don't touch.
        attention = tf.matmul(qkt, v)
        attention = self.out_reshape(attention)
        return attention


class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, filters, weight_decay_const, apply_batchnorm, padding='same'):

        super(ConvBlock, self).__init__()
        self.apply_batchnorm = apply_batchnorm
        self.c1 = Conv2D(filters=filters, kernel_size=3, strides=1, padding=padding,
                kernel_regularizer=l2(weight_decay_const))
        self.c2 = Conv2D(filters=filters, kernel_size=3, strides=1, padding=padding,
                kernel_regularizer=l2(weight_decay_const))
        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.activation = Activation('relu')

    def __call__(self, inputs):
        x = self.c1(inputs)
        if self.apply_batchnorm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.c2(x)
        if self.apply_batchnorm:
            x = self.bn2(x)
        return self.activation(x)

class UnetDownsample(tf.keras.layers.Layer):

    def __init__(self, filters, weight_decay_const, apply_batchnorm):

        super(UnetDownsample, self).__init__()

        self.cb1 = ConvBlock(filters, weight_decay_const, apply_batchnorm)
        self.cb2 = ConvBlock(filters*2, weight_decay_const, apply_batchnorm)
        self.cb3 = ConvBlock(filters*4, weight_decay_const, apply_batchnorm)
        self.cb4 = ConvBlock(filters*8, weight_decay_const, apply_batchnorm)
        self.cb5 = ConvBlock(filters*16, weight_decay_const, apply_batchnorm)
    
        self.max_pool = MaxPooling2D(pool_size=2, strides=2)

    def __call__(self, inputs):

        x1 = self.cb1(inputs)
        mp1 = self.max_pool(x1)

        x2 = self.cb2(mp1)
        mp2 = self.max_pool(x2)

        x3 = self.cb3(mp2)
        mp3 = self.max_pool(x3)

        x4 = self.cb4(mp3)
        mp4 = self.max_pool(x4)

        x5 = self.cb5(mp4)

        return [x1, x2, x3, x4, x5]


def unet_attention(input_shape, initial_filters, timesteps, n_classes, 
        weight_decay_const, apply_batchnorm):

    i1 = Input(input_shape)
    i2 = Input(input_shape)
    i3 = Input(input_shape)
    i4 = Input(input_shape)
    i5 = Input(input_shape)
    i6 = Input(input_shape)
     
    # now, apply embedding 
    inputs = [i1, i2, i3, i4, i5, i6]
    embeddings = []
    downsampler = UnetDownsample(initial_filters, weight_decay_const, apply_batchnorm)
    for inp in inputs:
        embeddings.append(downsampler(inp))

    filters_temp_attn_1 = 6
    filters_temp_attn_2 = 12
    temp_attn_1 = TemporalAttention(timesteps, (None, 32, 32, filters_temp_attn_1), 
            weight_decay_const)
    temp_attn_2 = TemporalAttention(timesteps, (None, 16, 16, filters_temp_attn_2),
            weight_decay_const)

    attn1_inputs = []
    attn2_inputs = []

    for e in embeddings:
        attn1_inputs.append(e[-2])
        attn2_inputs.append(e[-1])

    # TODO: Right now I have a really severe bottleneck layer!
    # I need to a). add residual connections (hmm, maybe not, b/c the other unet doesn't
    # use resid. connections.

    concatentated = []
    for i in range(timesteps-3):
        ls = []
        for j in range(timesteps):
            ls.append(embeddings[j][i])
        concatentated.append(Concatenate()(ls))

    dim_red_1 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same',
            kernel_regularizer=l2(weight_decay_const))(concatentated[0]) # 256x256x64

    dim_red_2 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same',
            kernel_regularizer=l2(weight_decay_const))(concatentated[1]) # 128x128x64

    dim_red_3 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same',
            kernel_regularizer=l2(weight_decay_const))(concatentated[2]) # 64x64x64

    query_embed_1 = Conv2D(filters=filters_temp_attn_1, kernel_size=1, strides=1, padding='same',
            kernel_regularizer=l2(weight_decay_const)) 
    value_embed_1 = Conv2D(filters=filters_temp_attn_1, kernel_size=1, strides=1, padding='same',
            kernel_regularizer=l2(weight_decay_const))
    query_embed_2 = Conv2D(filters=filters_temp_attn_2, kernel_size=1, strides=1, padding='same',
            kernel_regularizer=l2(weight_decay_const)) 
    value_embed_2 = Conv2D(filters=filters_temp_attn_2, kernel_size=1, strides=1, padding='same',
            kernel_regularizer=l2(weight_decay_const)) 

    query1 = []
    query2 = []
    value1 = []
    value2 = []

    for a1, a2 in zip(attn1_inputs, attn2_inputs):
        query1.append(Activation('relu')(query_embed_1(a1)))
        query2.append(Activation('relu')(query_embed_2(a2)))
        value1.append(Activation('relu')(value_embed_1(a1)))
        value2.append(Activation('relu')(value_embed_2(a2)))

    attention1, bs1 = temp_attn_1(query1, value1) # 32x32
    attention2, _ = temp_attn_2(query2, value2) # 16x16

    attention2 = convblock(attention2, initial_filters*16, weight_decay_const, apply_batchnorm)
    up1 = UpSampling2D(size=(2, 2))(attention2) # 32x32

    attention1 = convblock(attention1, initial_filters*16, weight_decay_const, apply_batchnorm)
    concat1 = Concatenate()([up1, attention1]) # 32x32

    x = convblock(concat1, initial_filters*8, weight_decay_const, apply_batchnorm)

    x = UpSampling2D(size=(2, 2))(x) # 64x64
    x = convblock(x, initial_filters*4, weight_decay_const, apply_batchnorm)
    x = Concatenate()([x, dim_red_3])
    x = convblock(x, initial_filters*4, weight_decay_const, apply_batchnorm)

    x = UpSampling2D(size=(2, 2))(x) # 128x128
    x = convblock(x, initial_filters*2, weight_decay_const, apply_batchnorm)
    x = Concatenate()([x, dim_red_2])
    x = convblock(x, initial_filters*2, weight_decay_const, apply_batchnorm)

    x = UpSampling2D(size=(2, 2))(x)
    x = convblock(x, initial_filters, weight_decay_const, apply_batchnorm)
    x = Concatenate()([x, dim_red_1])
    x = convblock(x, initial_filters, weight_decay_const, apply_batchnorm)
    softmax = Conv2D(n_classes, kernel_size=1, strides=1,
                        activation='softmax',
                        kernel_regularizer=l2(weight_decay_const))(x)
    return Model(inputs=[i1, i2, i3, i4, i5, i6], outputs=[softmax, bs1])



if __name__ == '__main__':
    
    '''
    shape = (16, 64, 64, 1)
    timesteps = 6
    queries = []
    values = []
    weight_decay_const = 0.0001
    for _ in range(timesteps):
        queries.append(tf.random.normal(shape))
        values.append(tf.random.normal(shape))
    layer = TemporalAttention(timesteps, shape, weight_decay_const)
    attention = layer(queries, values)
    '''
    model = unet_attention(input_shape=(256, 256, 6), 
                           initial_filters=16,
                           timesteps=6,
                           n_classes=3,
                           weight_decay_const=0.0001,
                           apply_batchnorm=True)
    # tensors = []
    # timesteps = 6
    # for _ in range(timesteps):
    #     tensors.append(tf.random.normal((4, 256, 256, 6)))
    # print(model.predict(tensors))
    print(model.summary(line_length=150))
