"""
Wasserstein GAN
https://arxiv.org/abs/1701.07875
"""
from tensorflow import keras
from tensorflow.keras import layers

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
        ], name='Discriminator')
        return model