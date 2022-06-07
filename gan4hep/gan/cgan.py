"""
This is a simple MLP-base conditional GAN.
Same as gan.py except that the conditional input is
given to the discriminator.
"""
from tensorflow import keras
from tensorflow.keras import layers

class CGAN():
    def __init__(self,
        noise_dim: int = 4, gen_output_dim: int = 2,
        cond_dim: int = 4, name="CGAN", **kwargs):
        """
        noise_dim: dimension of the noises
        gen_output_dim: output dimension
        cond_dim: in case of conditional GAN, 
                  it is the dimension of the condition
        """
        self.noise_dim = noise_dim
        self.gen_output_dim = gen_output_dim
        self.cond_dim = cond_dim

        self.gen_input_dim = self.noise_dim + self.cond_dim

        # Build the critic
        self.discriminator = self.build_critic()
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        self.name = name


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