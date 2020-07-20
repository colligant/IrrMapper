import os
import tensorflow.keras as keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

import utils



class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps satellite image patches to a feature representation """

    def __init__(self, input_shape, latent_dim, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        base = 8
        self.c1 = layers.Conv2D(base, kernel_size=3, activation='relu',
                padding='same', input_shape=input_shape)
        self.c11 = layers.Conv2D(base, kernel_size=3, activation='relu',
                padding='same')
        self.c2 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')
        self.c22 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')
        self.d1 = layers.MaxPooling2D()
        self.c3 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')
        self.c33 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')
        self.d2 = layers.MaxPooling2D()
        self.c4 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')
        self.c44 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(2048, activation='relu')
        self.fc2 = layers.Dense(1024, activation='relu')
        self.mean = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        self.reshape = layers.Reshape(target_shape=(32, 32, 6)) # hardcoded

    def call(self, inputs):
        print(inputs.shape)
        x = self.flatten(inputs)
        x = self.reshape(x)
        print(x.shape)
        x = self.c2(x)
        x = self.d1(x)
        x = self.c3(x)
        x = self.d2(x)
        x = self.c4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, output_channels, latent_dim, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        base = 8
        self.fc1 = layers.Dense(latent_dim, activation='relu')
        self.fc2 = layers.Dense(latent_dim*2, activation='relu')
        self.c1 = layers.Conv2D(base*2, kernel_size=3, activation='relu', 
                padding='same')
        self.c2 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')
        self.u1 = layers.UpSampling2D()
        self.c3 = layers.Conv2D(base, kernel_size=3, activation='relu',
                padding='same')
        self.u2 = layers.UpSampling2D()
        self.c4 = layers.Conv2D(base, kernel_size=3, activation='relu',
                padding='same')
        self.cout = layers.Conv2D(output_channels, kernel_size=1)
        self.reshape = layers.Reshape(target_shape=(8, 8, 32)) # hardcoded

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.reshape(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.u1(x)
        x = self.c3(x)
        x = self.u2(x)
        x = self.c4(x)
        return self.cout(x)


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        latent_dim,
        input_shape,
        output_channels,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(input_shape=input_shape, latent_dim=latent_dim)
        self.decoder = Decoder(output_channels=output_channels, latent_dim=latent_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


if __name__ == '__main__':
    
    latent_dim = 512
    input_shape = (None, None, 6)
    vae = VariationalAutoEncoder(latent_dim=latent_dim, input_shape=input_shape,
            output_channels=6)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()

    loss_metric = tf.keras.metrics.Mean()

    train_dataset = utils.make_test_dataset('/home/thomas/ssd/test-masked-with-points/')
    epochs = 5
    # Iterate over epochs.
    if not os.path.isfile('bs.pb.index'):
        for epoch in range(epochs):
            print("Start of epoch %d" % (epoch+1,))
            # Iterate over the batches of the dataset.
            for step, x_batch_train in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    reconstructed = vae(x_batch_train)
                    # Compute reconstruction loss
                    loss = mse_loss_fn(x_batch_train, reconstructed)
                    loss += sum(vae.losses)  # Add KLD regularization loss
                    # print(loss)

                grads = tape.gradient(loss, vae.trainable_weights)
                optimizer.apply_gradients(zip(grads, vae.trainable_weights))

                loss_metric(loss)

                if step % 100 == 0:
                    print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

        vae.save_weights('bs.pb')
    else:
        vae.load_weights('bs.pb')
    for x_batch in train_dataset:
        generated = vae(x_batch)
        for i in range(x_batch.shape[0]):
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(generated[i, :, :, :3] / np.max(generated[i, :, :, :3]))
            ax[1].imshow(x_batch[i, :, :, :3] / np.max(x_batch[i, :, :, :3]))
            plt.show()
