import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from gan4hep.utils_gan import log_metrics
# %%
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
        return self.decode(eps, apply_sigmoid=False)

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

# %%
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2*np.pi)
    return tf.reduce_sum(
        -0.5* ((sample - mean)**2 * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    fedality_loss = tf.compat.v1.losses.absolute_difference(x, x_logit)
    logpx_z = -tf.reduce_sum(fedality_loss)
    # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(x_logit, x)
    # logpx_z = -tf.reduce_sum(cross_ent)
    logpz = log_normal_pdf(z, 0., 0.)
    logpz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logpz_x)

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train(train_dataset, val_dataset, log_dir='log_training',
        latent_dim=2, lr=1e-4, epochs=10, **kwargs):
    optimizer = tf.keras.optimizers.Adam(lr)

    num_examples_to_generate = 16
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])

    model = VAE(latent_dim)

    summary_dir = os.path.join(log_dir, "logs")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    img_dir = os.path.join(log_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    def generate_and_save_images(model, epoch, datasets, **kwargs):
        predictions = []
        truths = []
        for data in datasets:
            _, test_truth = data
            test_truth = (test_truth + 1) * 0.5
            mean, logvar = model.encode(test_truth)
            z = model.reparameterize(mean, logvar)
            predictions.append(model.sample(z))
            truths.append(test_truth)

        predictions = tf.concat(predictions, axis=0).numpy()
        truths = tf.concat(truths, axis=0).numpy()

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        axs = axs.flatten()

        config = dict(histtype='step', lw=2)
        # phi
        idx=0
        ax = axs[idx]
        x_range = [-1, 1]
        yvals, _, _ = ax.hist(
            truths[:, idx], bins=40, range=x_range, label='Truth', **config)
        max_y = np.max(yvals) * 1.1
        ax.hist(
            predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
        ax.set_xlabel(r"$\phi$")
        ax.set_ylim(0, max_y)
        
        # theta
        idx=1
        ax = axs[idx]
        yvals, _, _ = ax.hist(truths[:, idx],  bins=40, range=x_range, label='Truth', **config)
        max_y = np.max(yvals) * 1.1
        ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
        ax.set_xlabel(r"$theta$")
        ax.set_ylim(0, max_y)

        # plt.legend()
        plt.savefig(os.path.join(img_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
        plt.close('all')
        return log_metrics(summary_writer, predictions, truths, epoch, **kwargs)[0]

    best_metric = 999
    for epoch in range(1, epochs+1):
        for data in train_dataset:
            _, train_truth = data
            train_truth = (train_truth + 1) * 0.5
            train_step(model, train_truth, optimizer)

        loss = generate_and_save_images(model, epoch, val_dataset)        

        best_metric = min(best_metric, loss)
        print('Epoch: {}, BEST ELBO: {}'.format(epoch, best_metric))

# %%
if __name__ == '__main__':
    import argparse
    from gan4hep.preprocess import herwig_angles


    parser = argparse.ArgumentParser(description='Train VAE')
    add_arg = parser.add_argument
    add_arg('filename', help='input filename')
    add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    add_arg("--latent-dim", default=2, type=int, help='latent dimension')
    add_arg("--epochs", help='number of epochs', type=int, default=10)
    add_arg("--log-dir", help='log directory', default='log_training')
    args = parser.parse_args()

    batch_size = args.batch_size

    
    train_in, train_truth, test_in, test_truth = herwig_angles(
        args.filename, max_evts=args.max_evts)

    AUTO = tf.data.experimental.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_in, train_truth)).shuffle(2*batch_size).batch(
            batch_size, drop_remainder=True).prefetch(AUTO)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (test_in, test_truth)).shuffle(2*batch_size).batch(
            batch_size, drop_remainder=True).prefetch(AUTO)

    train(train_dataset, val_dataset, **vars(args))