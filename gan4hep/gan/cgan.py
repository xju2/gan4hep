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

        layer_size = 256
        num_layers = 10
        layer_list = [keras.Input(shape=(gen_input_dim,))]
        for _ in range(num_layers):
            layer_list += [
                layers.Dense(layer_size),
                layers.LayerNormalization(),
                layers.Activation("tanh"),
            ]
        layer_list += [layers.Dense(self.gen_output_dim), layers.Activation("tanh")]
        model = keras.Sequential(layer_list, name='Generator')

        return model

    def build_critic(self):
        gen_output_dim = self.gen_output_dim + self.cond_dim

        layer_size = 256
        num_layers = 10
        layer_list = [keras.Input(shape=(gen_output_dim,))]
        for _ in range(num_layers):
            layer_list += [
                layers.Dense(layer_size),
                layers.LayerNormalization(),
                layers.LeakyReLU(),
            ]
        layer_list += [layers.Dense(1, activation='sigmoid')]
        model = keras.Sequential(layer_list, name='Discriminator')
        return model