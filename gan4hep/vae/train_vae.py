#!/usr/bin/env python

"""
This script trains a MLP-based GAN that generates `\phi` and `\theta'
of the two outgoing particles
"""

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# from IPython import display

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import tqdm
from gan4hep.utils_gan import log_metrics

from tensorflow.compat.v1 import logging #

class VAE(tf.keras.Model):
    """Variational autoencoder"""

    def __init__(self, latent_dim, input_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = keras.Sequential([
            keras.Input(shape=(input_dim,)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            
            layers.Dense(latent_dim+latent_dim),
        ])

        self.decoder = keras.Sequential([
            keras.Input(shape=(latent_dim,)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            
            layers.Dense(input_dim),            
        ])

        @tf.function
        def sample(self, eps=None):
            if eps is None:
                eps = tf.random.normal(shape=(100, self.latent_dim))
            return self.decode(eps, apply_sigmoid=True)

        def encode(self, x):
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
            return mean, logvar

        def reparameterize(self, mean, logvar):
            eps = tf.random.normal(shape=mean.shape)
            return eps * tf.exp(logvar * .5) + mean

        def decode(self, z, apply_sigmoid=False):
            logits = self.decoder(z)
            if apply_sigmoid:
                probs = tf.sigmoid(logits)
                return probs
            return logits

def run_training(
    filename, noise_dim=4, num_test_evts=5000,
    max_evts=None, batch_size=512, gen_output_dim=2,
    epochs=50, lr=1e-4,
    log_dir='log_training', inference=False,
    **kwargs
    ):

    logging.info("TF Version:{}".format(tf.__version__))
    # read input datasets
    df = pd.read_csv(filename, sep=';', 
                header=None, names=None, engine='python')

    event = None
    with open(filename, 'r') as f:
        for line in f:
            event = line
            break
    particles = event[:-2].split(';')

    input_4vec = df[0].str.split(",", expand=True)[[4, 5, 6, 7]].to_numpy().astype(np.float32)
    out_particles = []
    for idx in range(1, len(particles)):
        out_4vec = df[idx].str.split(",", expand=True).to_numpy()[:, -4:].astype(np.float32)
        out_particles.append(out_4vec)

    out_4vec = out_particles[0]
    px = out_4vec[:, 1].astype(np.float32)
    py = out_4vec[:, 2].astype(np.float32)
    pz = out_4vec[:, 3].astype(np.float32)
    pT = np.sqrt(px**2 + py**2)
    phi = np.arctan(px/py)
    theta = np.arctan(pT/pz)
    truth_in = np.stack([phi, theta], axis=1)

    # convert them to datasets

    logging.info("Input events: {:,}".format(input_4vec.shape[0]))
    AUTO = tf.data.experimental.AUTOTUNE

    # testing data
    # the training data will be composed for each epoch
    test_in = input_4vec[:num_test_evts]
    noise = np.random.normal(loc=0., scale=1., size=(test_in.shape[0], noise_dim))
    test_in = np.concatenate([test_in, noise], axis=1).astype(np.float32)
    test_truth = truth_in[:num_test_evts]
    testing_data = tf.data.Dataset.from_tensor_slices(
        (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)


    train_in = input_4vec[num_test_evts:max_evts]
    train_truth = truth_in[num_test_evts:max_evts]

    # ===========
    # optimizer
    # ===========
    generator_optimizer = keras.optimizers.Adam(lr_gen)
    discriminator_optimizer = keras.optimizers.Adam(lr_disc)

    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
    logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

    summary_dir = os.path.join(log_dir, "logs")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    img_dir = os.path.join(log_dir, 'img')
    os.makedirs(img_dir, exist_ok=True)
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(gen_in_4vec, truth_4vec):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_out_4vec = generator(gen_in_4vec, training=True)

            real_output = discriminator(truth_4vec, training=True)
            fake_output = discriminator(gen_out_4vec, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return disc_loss, gen_loss


    def train(epochs):
        best_wdis = 900
        for epoch in tqdm.trange(epochs):
            start = time.time()
            
            # compose the training dataset by generating different noises
            noise = np.random.normal(loc=0., scale=1., size=(train_in.shape[0], noise_dim))
            train_inputs = np.concatenate([train_in, noise], axis=1).astype(np.float32)
            dataset = tf.data.Dataset.from_tensor_slices(
                (train_inputs, train_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

            tot_loss = []
            for data_batch in dataset:
                tot_loss.append(list(train_step(*data_batch)))
            
            tot_loss = np.array(tot_loss)
            avg_loss = np.sum(tot_loss, axis=0)/tot_loss.shape[0]
            loss_dict = dict(disc_loss=avg_loss[0], gen_loss=avg_loss[1])
            # display.clear_output(wait=True)
            tot_wdis = generate_and_save_images(generator, epoch+1, testing_data, **loss_dict)
            if tot_wdis < best_wdis:
                ckpt_manager.save()
                generator.save("generator")
                best_wdis = tot_wdis

    def generate_and_save_images(model, epoch, datasets, **kwargs) -> float:
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = []
        truths = []
        for data in datasets:
            test_input, test_truth = data
            predictions.append(model(test_input, training=False))
            truths.append(test_truth)

        predictions = tf.concat(predictions, axis=0).numpy()
        truths = tf.concat(truths, axis=0).numpy()

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        axs = axs.flatten()

        config = dict(histtype='step', lw=2)
        # phi
        idx=0
        ax = axs[idx]
        ax.hist(truths[:, idx], bins=40, range=[-np.pi, np.pi], label='Truth', **config)
        ax.hist(predictions[:, idx], bins=40, range=[-np.pi, np.pi], label='Generator', **config)
        ax.set_xlabel(r"$\phi$")
        ax.set_ylim(0, 450* num_test_evts/5000)
        
        # theta
        idx=1
        ax = axs[idx]
        ax.hist(truths[:, idx],  bins=40, range=[-2, 2], label='Truth', **config)
        ax.hist(predictions[:, idx], bins=40, range=[-2, 2], label='Generator', **config)
        ax.set_xlabel(r"$theta$")
        ax.set_ylim(0, 450*num_test_evts/5000)
        
        # plt.legend()
        plt.savefig(os.path.join(img_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.close('all')

        return log_metrics(summary_writer, predictions, truths, epoch, **kwargs)[0]

    
    if not inference:
        train(epochs)
    else:
        logging.info("Performing inference only")
        generate_and_save_images(generator, 0, testing_data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("filename", help='input filename', default=None)
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--lr-gen", help='learning rate for generator', default=1e-4, type=float)
    add_arg("--lr-disc", help='learning rate for discriminator', default=1e-4, type=float)
    add_arg("--num-test-evts", help='number of testing events', default=5000, type=int)
    add_arg("--inference", help='perform inference only', action='store_true')
    add_arg("-v", '--verbose', help='tf logging verbosity', default='ERROR',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    args = parser.parse_args()

    logging.set_verbosity(args.verbose)
    if args.filename and os.path.exists(args.filename):
        run_training(**vars(args))