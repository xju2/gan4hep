"""
Adversarial Autoencoders
https://arxiv.org/abs/1511.05644
"""
import numpy as np
import os


import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))
gpus = tf.config.experimental.list_physical_devices("GPU")
logging.info("found {} GPUs".format(len(gpus)))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from tensorflow import keras
from tensorflow.keras import layers

import tqdm


cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return tf.reduce_mean(total_loss)

def generator_loss(fake_output):
    return tf.reduce_mean(cross_entropy(tf.ones_like(fake_output), fake_output))


class AdversarialAutoencoder():
    def __init__(self,
        noise_dim: int = 4, gen_output_dim: int = 2,
        cond_dim: int = 4, disable_tqdm=False, lr=0.0001):
        """
        noise_dim: dimension of the noises
        gen_output_dim: output dimension
        cond_dim: in case of conditional GAN, 
                  it is the dimension of the condition
        """
        self.noise_dim = noise_dim
        self.gen_output_dim = gen_output_dim
        self.cond_dim = cond_dim
        self.disable_tqdm = disable_tqdm

        self.gen_input_dim = self.noise_dim + self.cond_dim

        # ============
        # Optimizers
        # ============
        self.generator_optimizer = keras.optimizers.Adam(lr)
        self.discriminator_optimizer = keras.optimizers.Adam(lr)

        # Build the critic
        self.discriminator = self.build_critic()
        self.discriminator.summary()

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        


    def build_encoder(self):
        gen_input_dim = self.gen_input_dim

        model = keras.Sequential([
            keras.Input(shape=(gen_input_dim,)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            
            layers.Dense(self.gen_output_dim),
            layers.Activation("tanh"),
        ], name='Generator')
        return model

    def build_critic(self):
        gen_output_dim = self.gen_output_dim + self.cond_dim

        model = keras.Sequential([
            keras.Input(shape=(gen_output_dim,)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Dense(1, activation='sigmoid'),
        ], name='Discriminator')
        return model


    def train(self, train_truth, epochs, batch_size, test_truth, log_dir, evaluate_samples_fn,
        train_in=None, test_in=None):
        # ======================================
        # construct testing data once for all
        # ======================================
        AUTO = tf.data.experimental.AUTOTUNE
        noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], self.noise_dim))
        test_in = np.concatenate(
            [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise


        testing_data = tf.data.Dataset.from_tensor_slices(
            (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

        # ====================================
        # Checkpoints and model summary
        # ====================================
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
        logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
        _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

        summary_dir = os.path.join(log_dir, "logs")
        summary_writer = tf.summary.create_file_writer(summary_dir)

        img_dir = os.path.join(log_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)

        @tf.function
        def train_step(gen_in_4vec, truth_4vec):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_out_4vec = self.generator(gen_in_4vec, training=True)

                real_output = self.discriminator(truth_4vec, training=True)
                fake_output = self.discriminator(gen_out_4vec, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

            return disc_loss, gen_loss

        best_wdis = 9999
        with tqdm.trange(epochs, disable=self.disable_tqdm) as t0:
            for epoch in t0:

                # compose the training dataset by generating different noises for each epochs
                noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], self.noise_dim))
                train_inputs = np.concatenate(
                    [train_in, noise], axis=1).astype(np.float32) if train_in is not None else noise


                dataset = tf.data.Dataset.from_tensor_slices(
                    (train_inputs, train_truth)).shuffle(2*batch_size).batch(batch_size, drop_remainder=True).prefetch(AUTO)

                tot_loss = []
                for data_batch in dataset:
                    tot_loss.append(list(train_step(*data_batch)))

                tot_loss = np.array(tot_loss)
                avg_loss = np.sum(tot_loss, axis=0)/tot_loss.shape[0]
                loss_dict = dict(D_loss=avg_loss[0], G_loss=avg_loss[1])
                t0.set_postfix(**loss_dict)

                tot_wdis = evaluate_samples_fn(self.generator, epoch, testing_data, summary_writer, img_dir, **loss_dict)
                if tot_wdis < best_wdis:
                    ckpt_manager.save()
                    self.generator.save("generator")
                    best_wdis = tot_wdis


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("filename", help='input filename', default=None)
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--num-test-evts", help='number of testing events', default=5000, type=int)
    add_arg("--inference", help='perform inference only', action='store_true')
    add_arg("-v", '--verbose', help='tf logging verbosity', default='ERROR',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    args = parser.parse_args()

    logging.set_verbosity(args.verbose)


    from gan4hep.utils_gan import generate_and_save_images
    from gan4hep.preprocess import herwig_angles

    train_in, train_truth, test_in, test_truth = herwig_angles(args.filename, args.max_evts)

    batch_size = args.batch_size
    gan = GAN()
    gan.train(
        train_truth, args.epochs, batch_size,
        test_truth, args.log_dir,
        generate_and_save_images,
        train_in, test_in
    )