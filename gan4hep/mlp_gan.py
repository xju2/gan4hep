"""
MLP-based GANs
"""
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt

from gan4hep.gan_base import GANBase

class Generator(snt.Module):
    def __init__(self, latent_size=512, num_layers=10,
                activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
                name="Generator"
        ):
        super().__init__(name=name)

        self._core = snt.nets.MLP(
            [latent_size]*num_layers+[8], activation=activation,
            activate_final=False, dropout_rate=None,
            name="core_generator")

    def __call__(self,
                input_op,
                is_training: bool = True) -> tf.Tensor:
        """
        Args: 
            input_op: 2D vector with dimensions [batch-size, features], 
        Retruns:
            predicted node featurs with dimension of [batch-size, out-features] 
        """
        return tf.nn.tanh(self._core(input_op))


class Discriminator(snt.Module):
    def __init__(self,
                latent_size=512, num_layers=10,
                activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
                name="Discriminator"
        ):
        super().__init__(name=name)
        self._core = snt.nets.MLP(
            [latent_size]*num_layers+[1], activation=activation,
            activate_final=False, dropout_rate=None,
            name="core_discriminator"
            )


    def __call__(self, input_op, is_training=True):
        return self._core(input_op)



class GAN(GANBase):
    def __init__(self, noise_dim, batch_size, latent_size=512, num_layers=10, name=None):
        super().__init__(noise_dim, batch_size, name=name)
        self.generator = Generator(latent_size=latent_size, num_layers=num_layers)
        self.discriminator = Discriminator(latent_size=latent_size, num_layers=num_layers)