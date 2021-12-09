"""
Wasserstein GAN
https://arxiv.org/abs/1701.07875
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

class WGAN():
    def __init__(self,
        noise_dim: int = 4, gen_output_dim: int = 2,
        cond_dim: int = 4, disable_tqdm=False):
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

        # some pre-defined settings
        self.n_critics = 5
        self.lr = 0.00005
        self.clip_value = 0.01

        self.gen_input_dim = self.noise_dim + self.cond_dim

        optimizer = keras.optimizers.RMSprop(lr=self.lr)
        # Build the critic
        self.discriminator = self.build_critic()
        self.discriminator.compile(
            loss=self.wasserstein_loss,
            optimizer=optimizer
        )
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Now combine generator and critic
        z = keras.Input(shape=(self.gen_input_dim,))
        particles = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(particles)
        self.combined = keras.Model(z, valid, name='Combined')
        self.combined.compile(
            loss=self.wasserstein_loss,
            optimizer=optimizer,
        )
        self.combined.summary()

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def build_generator(self):
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
        gen_output_dim = self.gen_output_dim

        model = keras.Sequential([
            keras.Input(shape=(gen_output_dim,)),
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            
            layers.Dense(256),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            layers.Dense(1, activation='sigmoid'),
        ], name='Critic')
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

        best_wdis = 9999
        best_epoch = -1
        with tqdm.trange(epochs, disable=self.disable_tqdm) as t0:
            for epoch in t0:

                # compose the training dataset by generating different noises
                noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], self.noise_dim))
                train_inputs = np.concatenate(
                    [train_in, noise], axis=1).astype(np.float32) if train_in is not None else noise


                dataset = tf.data.Dataset.from_tensor_slices(
                    (train_inputs, train_truth)).shuffle(2*batch_size).batch(batch_size, drop_remainder=True).prefetch(AUTO)

                valid = -np.ones((batch_size, 1))
                fake  = np.ones((batch_size, 1))

                tot_loss = []
                icritic = 0
                for data_batch in dataset:

                    #---------------------
                    # Training Discriminator
                    #---------------------
                    gen_in, truth = data_batch
                    gen_out = self.generator.predict(gen_in)
                    d_loss_real = self.discriminator.train_on_batch(truth, valid)
                    d_loss_fake = self.discriminator.train_on_batch(gen_out, fake)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                    # clip critic weights
                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [tf.clip_by_value(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)
                    icritic += 1
                    if icritic < self.n_critics:
                        continue

                    #-----------------
                    # Training Generator
                    #-----------------
                    g_loss = self.combined.train_on_batch(gen_in, valid)

                    tot_loss.append([d_loss, g_loss])

                tot_loss = np.array(tot_loss)
                avg_loss = np.sum(tot_loss, axis=0)/tot_loss.shape[0]
                loss_dict = dict(D_loss=avg_loss[0], G_loss=avg_loss[1])

                tot_wdis = evaluate_samples_fn(self.generator, epoch, testing_data, summary_writer, img_dir, **loss_dict)
                if tot_wdis < best_wdis:
                    ckpt_manager.save()
                    self.generator.save("generator")
                    best_wdis = tot_wdis
                    best_epoch = epoch
                t0.set_postfix(**loss_dict, BestD=best_wdis, BestE=best_epoch)
        tmp_res = "Best Model in {} Epoch with a Wasserstein distance {:.4f}".format(best_epoch, best_wdis)
        logging.info(tmp_res)
        summary_logfile = os.path.join(summary_dir, 'results.txt')
        with open(summary_logfile, 'a') as f:
            f.write(tmp_res + "\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("filename", help='input filename', default=None)
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--num-test-evts", help='number of testing events', default=10000, type=int)
    add_arg("--inference", help='perform inference only', action='store_true')
    add_arg("-v", '--verbose', help='tf logging verbosity', default='INFO',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    args = parser.parse_args()

    logging.set_verbosity(args.verbose)


    from gan4hep.utils_gan import generate_and_save_images
    from gan4hep.preprocess import herwig_angles

    train_in, train_truth, test_in, test_truth = herwig_angles(args.filename, args.max_evts)

    batch_size = args.batch_size
    gan = WGAN()
    gan.train(
        train_truth, args.epochs, batch_size,
        test_truth, args.log_dir,
        generate_and_save_images,
        train_in, test_in
    )