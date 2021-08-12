"""
MLP predicts N number of output values
"""
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tqdm

class WGAN():
    def __init__(self,
        noise_dim: int, gen_output_dim: int,
        cond_dim: int = 0, disable_tqdm=False):
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
        self.critic = self.build_critic()
        self.critic.compile(
            loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy']
        )
        self.critic.summary()

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Now combine generator and critic
        z = keras.Input(shape=(self.gen_input_dim,))
        particles = self.generator(z)

        self.critic.trainable = False

        valid = self.critic(particles)
        self.combined = keras.Model(z, valid, name='Combined')
        self.combined.compile(
            loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy']
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
        train_in=None, test_in=None, sample_interval=50):
        # ======================================
        # construct testing data once for all
        # ======================================
        AUTO = tf.data.experimental.AUTOTUNE
        noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], self.noise_dim))
        if test_in is not None:
            test_in = np.concatenate([test_in, noise], axis=1).astype(np.float32)
        else:
            test_in = noise

        testing_data = tf.data.Dataset.from_tensor_slices(
            (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

        # ====================================
        # Checkpoints and model summary
        # ====================================
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.critic)
        ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
        logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
        _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

        summary_dir = os.path.join(log_dir, "logs")
        summary_writer = tf.summary.create_file_writer(summary_dir)

        img_dir = os.path.join(log_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)

        best_wdis = 9999
        with tqdm.trange(epochs, disable=self.disable_tqdm) as t0:
            for epoch in t0:
                t0.set_description('Epoch {}/{}'.format(epoch, epochs))

                # compose the training dataset by generating different noises
                noise = np.random.normal(loc=0., scale=1., size=(train_truth.shape[0], self.noise_dim))
                if train_in is not None:
                    train_inputs = np.concatenate([train_in, noise], axis=1).astype(np.float32)
                else:
                    train_inputs = noise

                dataset = tf.data.Dataset.from_tensor_slices(
                    (train_inputs, train_truth)).shuffle().batch(batch_size, drop_remainder=True).prefetch(AUTO)

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
                    d_loss_real = self.critic.train_on_batch(truth, valid)
                    d_loss_fake = self.critic.train_on_batch(gen_out, fake)
                    d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                    # clip critic weights
                    for l in self.critic.layers:
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
                loss_dict = dict(disc_loss=avg_loss[0], gen_loss=avg_loss[1])

                tot_wdis = evaluate_samples_fn(self.generator, epoch, testing_data, summary_writer, img_dir, **loss_dict)
                if tot_wdis < best_wdis:
                    ckpt_manager.save()
                    self.generator.save("generator")
                    best_wdis = tot_wdis
                    
                t0.set_postfix(G_loss=g_loss, D_loss=d_loss)


if __name__ == '__main__':
    import argparse
    # parser = argparse.ArgumentParser(description='Test WGAN')
    # add_arg = parser.add_argument
    # add_arg('', help='')
    
    # args = parser.parse_args()

    from 
    gan = WGAN(8, 2, 4)
    