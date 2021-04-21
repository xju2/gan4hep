"""
RNN-based Generator and MLP-based Discriminator
"""
from typing import Callable, Iterable, Optional, Text

import tensorflow as tf
import sonnet as snt

from gan4hep.gan_base import GANBase

from gan4hep.mlp_gan import Discriminator
# from gan4hep.reader import n_max_nodes # maximum number of output particles


class Generator(snt.Module):
    def __init__(self, max_nodes=2,
        out_dim: int = 4,
        latent_size=512, num_layers=5,
        rnn_latent_size=512,
        activation: Callable[[tf.Tensor], tf.Tensor] = tf.nn.leaky_relu,
        name="Generator",
        ):
        """
        initilize the generator by giving the input dimension and output dimension.
        """
        super().__init__(name=name)

        self._node_linear = snt.nets.MLP(
            [latent_size]*num_layers+[out_dim], activation=activation,
            activate_final=False, dropout_rate=None,
            name="node_encoder")

        # deep RNN
        self._node_rnn = snt.DeepRNN([
            snt.LSTM(hidden_size=rnn_latent_size),
            snt.LSTM(hidden_size=rnn_latent_size)
        ])
        # self._node_rnn = snt.GRU(hidden_size=LATENT_SIZE, name="node_rnn")
        self._node_prop_nn = snt.nets.MLP(
            [latent_size]*num_layers+[out_dim], activation=activation,
            activate_final=False, dropout_rate=None,
            name="node_prop_nn")

        self.out_dim = out_dim
        self.max_nodes = max_nodes

    def __call__(self,
                input_op,
                training: bool = True) -> tf.Tensor:
        """
        Args: 
            input_op: 2D vector with dimensions [batch-size, features], 
            the latter containing [px, py, pz, E, N-dimension noises]
            max_nodes: maximum number of output nodes
            training: if in training mode, needed for `dropout`.

        Retruns:
            predicted node featurs with dimension of [batch-size, max-nodes, out-features] 
        """

        batch_size = input_op.shape[0]
        node_hidden_state = self._node_rnn.initial_state(batch_size)

        nodes = self._node_linear(input_op)

        nodes = tf.reshape(nodes, [batch_size, 1, self.out_dim])
        for inode in range(self.max_nodes):

            # node properties
            node_embedding = tf.math.reduce_sum(nodes, axis=1)
            node_embedding, node_hidden_state = self._node_rnn(
                node_embedding, node_hidden_state)
            node_prop = self._node_prop_nn(node_embedding)
            node_prop = tf.reshape(node_prop, [batch_size, 1, self.out_dim])

            # add new node to the existing nodes
            nodes = tf.concat([nodes, node_prop], axis=1, name='add_new_node')
        return tf.reshape(nodes[:, 1:, :], [batch_size, -1])


class GAN(GANBase):
    def __init__(
            self, noise_dim, batch_size,
            max_nodes=2, latent_size=512, num_layers=10, name=None):
        super().__init__(noise_dim, batch_size, name=name)
        self.generator = Generator(max_nodes=max_nodes, latent_size=latent_size, num_layers=num_layers)
        self.discriminator = Discriminator(latent_size=latent_size, num_layers=num_layers)